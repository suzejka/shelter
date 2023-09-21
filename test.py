def check_and_replace_color(colors):
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            color1 = colors[i]
            color2 = colors[j]
            reversed_color2 = "/".join(reversed(color2.split("/")))
            
            if color1 == reversed_color2:
                colors[j] = color1
    return colors

color_column = [
    "WHITE/BLACK", 
    "BLACK/WHITE", 
    "BROWN/BLACK", 
    "BLACK/BROWN"
]

updated_colors = check_and_replace_color(color_column)
print(updated_colors)