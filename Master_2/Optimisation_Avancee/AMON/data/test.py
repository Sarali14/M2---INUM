import shapefile
import matplotlib.pyplot as plt

# List your shapefiles
shapefiles = ["boundary_zone.shp", "exclusion_zones.shp"]

# Define colors for each shapefile
colors = ["blue", "red"]

for shp_file, color in zip(shapefiles, colors):
    sf = shapefile.Reader(shp_file)
    print(f"{shp_file} contains {len(sf.shapes())} shapes")
    print(f"\nCoordinates for {shp_file}:")
    first_shape = True  # To label each shapefile only once in the plot
    
    for i, shape in enumerate(sf.shapes(), start=1):
        points = shape.points
        if not points:
            print(f" Shape {i}: (no coordinates)")
            continue
        
        # Print coordinates
        print(f" Shape {i}:")
        for x, y in points:
            print(f"   ({x}, {y})")
        points = shape.points
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        label = shp_file if first_shape else None
        plt.plot(x, y, color=color, label=label)
        first_shape = False

# Optional: prevent duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Overlay of two shapefiles")
plt.show()
