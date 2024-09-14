import src.seg_utils as seg_utils

name = "barak1sdfsdfsdfsdf!@@$$%&sdfsdfsdfsdfsdfsdfsdf"

color = seg_utils.string_to_rgba_color(name,normalize=False)
print(color)

color = seg_utils.string_to_rgba_color(name,normalize=True)
print(color)

print([color * (2**16 - 1) for color in color])
