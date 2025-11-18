import shapefile
import matplotlib.pyplot as plt

# Scale factor you want to apply
scale_factor = 1.2

# List your shapefiles
shapefiles = ["poly2.shp"]

# Define colors for each shapefile
colors = ["blue"]

for shp_file, color in zip(shapefiles, colors):
    sf = shapefile.Reader(shp_file)
    print(f"{shp_file} contains {len(sf.shapes())} shapes")
    print(f"\nScaled coordinates for {shp_file}:")
    first_shape = True  # To label each shapefile only once in the plot
    
    for i, shape in enumerate(sf.shapes(), start=1):
        points = shape.points
        if not points:
            print(f" Shape {i}: (no coordinates)")
            continue
        
        print(f" Shape {i}:")
        x = []
        y = []

        # Print and store scaled coordinates
        for px, py in points:
            sx = scale_factor * px
            sy = scale_factor * py
            print(f"   ({sx}, {sy})")
            x.append(sx)
            y.append(sy)

        # Close polygon if needed
        if (points[0] != points[-1]):
            x.append(scale_factor * points[0][0])
            y.append(scale_factor * points[0][1])

        # Plot the scaled polygon
        label = shp_file if first_shape else None
        plt.plot(x, y, color=color, label=label)
        first_shape = False

# Optional: prevent duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel("X (scaled)")
plt.ylabel("Y (scaled)")
plt.title("Scaled shapefile polygon")
plt.show()
