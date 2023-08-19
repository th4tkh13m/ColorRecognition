import json

with open("colors.json") as f:
    colors = json.load(f)["colors"]
    
new_colors = dict()
for hex, name in colors:
    hex = "#" + hex
    
    new_colors[hex] = name

with open("newcolors.json", "w") as f:
    f.write(json.dumps(new_colors))
    