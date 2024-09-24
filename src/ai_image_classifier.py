import time, random
from collections import Counter
import numpy as np
import cv2
from nltk.corpus import wordnet
import torch
import torchvision.transforms.functional as TF
import open_clip
from ai_models.data.seed_words import seed_words
from src.constants import AI_MODEL
from src.object_classifier_utils import is_noun



class AIImageClassifier:
    def __init__(self, model_accuracy='s', max_labels=10000,seed=None):
        self.model_accuracy = model_accuracy
        self.max_labels = max_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.labels = None
        self.text_inputs = None
        self.raw_final_labels = []
        self.final_labels = {}
        if seed is not None:
            self.set_seed(seed)
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self._load_model()
        self._generate_labels()
    
    def set_seed(self, seed, cuda_deterministic=True):
 # Set seed for Python's random module
        random.seed(seed)
        
        # Set seed for NumPy's random generator
        np.random.seed(seed)
        
        # Set seed for PyTorch on CPU
        torch.manual_seed(seed)
        
        # Set seed for PyTorch on CUDA, if available
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Ensure deterministic behavior for CUDA
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _load_model(self):
        model_name = AI_MODEL[self.model_accuracy][0]
        pretrained_weights = AI_MODEL[self.model_accuracy][1]
        
        # Load model
        start_time = time.time()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_weights)
        self.model = self.model.to(self.device).half().eval()
        print(f"Model {model_name} loaded in {time.time() - start_time:.2f} seconds")
    
    def _generate_labels(self):
        self.labels = self.generate_noun_labels(seed_words, self.max_labels)
        self.text_inputs = self.tokenize_labels(self.labels, self.device)
    
    def generate_noun_labels(self, word_list, max_labels=10000):
        noun_labels = set()
        for word in word_list:
            synsets = wordnet.synsets(word, pos=wordnet.NOUN)
            for syn in synsets:
                noun_labels.update(lemma.name() for lemma in syn.lemmas())
                noun_labels.update(lemma.name() for lemma in syn.hypernyms())
                noun_labels.update(lemma.name() for lemma in syn.hyponyms())
            if len(noun_labels) >= max_labels:
                break
        return list(noun_labels)[:max_labels]
    
    def tokenize_labels(self, labels, device):
        return torch.cat([open_clip.tokenize(label) for label in labels]).to(device)
    
    def process_in_batches(self, image_features):
        max_prob = -1
        best_label = None
        batch_size = 500
        labels = self.labels
        text_inputs = self.text_inputs
        model = self.model

        for i in range(0, len(labels), batch_size):
            batch_labels = labels[i:i+batch_size]
            batch_text_inputs = text_inputs[i:i+batch_size]

            with torch.no_grad(), torch.amp.autocast("cuda"):
                text_features = model.encode_text(batch_text_inputs)
                logits_per_image = image_features @ text_features.T
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                batch_max_prob = probs.max()
                if batch_max_prob > max_prob:
                    max_prob = batch_max_prob
                    best_label = batch_labels[probs.argmax()]

            del text_features
            torch.cuda.empty_cache()
        return best_label, max_prob
    
    def preprocess_image(self, image_array):
        """
        Preprocesses an image array using torchvision's functional API without PIL.
        """
        # Convert BGR (OpenCV) to RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and scale pixel values to [0, 1]
        image_rgb = image_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        
        # Resize to 224x224
        image_tensor = TF.resize(image_tensor, [224, 224])
        
        # Normalize using mean and std
        image_tensor = TF.normalize(image_tensor,
                                    mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
        
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Convert to half precision if model is in half precision
    
        image_tensor = image_tensor.half()
    
        
        return image_tensor
    
    def encode_image(self, image_tensor):
        with torch.no_grad(), torch.cuda.amp.autocast():
            return self.model.encode_image(image_tensor)
    
    def process_image(self, image_array):

        try:
            image_tensor = self.preprocess_image(image_array)
            start_time = time.time()
            image_features = self.encode_image(image_tensor)
            #print(f"Image encoded in {time.time() - start_time:.2f} seconds")
            best_label, best_prob = self.process_in_batches(image_features)
            best_label = best_label.split('.')[0]
            #print(f"Predicted label: {best_label} with probability {best_prob:.4f}")
            return best_label
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def update_final_labels(self, best_label):
        if best_label:
            if best_label in self.final_labels:
                self.final_labels[best_label] += 1
            else:
                self.final_labels[best_label] = 1
    
    def get_best_label(self):
        if self.final_labels:
            max_label = max(self.final_labels, key=self.final_labels.get)
            
            
            if self.final_labels[max_label] < 2: # if we not "sure" the label is correct,we will return the most common word in the labels list
                
                return "undefined"

            return max_label
        else:
            print("No labels detected.")
            return "undefined"
    
    def classify_images(self, image_arrays)->None:
        """
        Classifies a list of OpenCV images (NumPy arrays) and returns the most frequent label.
        """
        self.final_labels = {}
        self.raw_final_labels = []
        
        for image_array in image_arrays:
            best_label = self.process_image(image_array)
            self.raw_final_labels.append(best_label)
            self.update_final_labels(best_label)
        
