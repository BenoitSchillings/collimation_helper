import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import optimize

def find_inner_edge1(img, center, angle, threshold):
    h, w = img.shape
    max_r = min(h, w) // 2
    x, y = center
    for r in range(max_r):
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        if px < 0 or px >= w or py < 0 or py >= h:
            return max_r
        if img[py, px] > threshold:
            return r
    return max_r


def find_inner_edge(img, center, angle, threshold):
    h, w = img.shape
    max_r = min(h, w) // 2
    x, y = center

    # Create a range of radii
    r_values = np.arange(max_r)

    # Calculate all possible (px, py) in one step
    px_values = (x + r_values * np.cos(angle)).astype(int)
    py_values = (y + r_values * np.sin(angle)).astype(int)

    # Filter out (px, py) values that are out of bounds
    valid_indices = (px_values >= 0) & (px_values < w) & (py_values >= 0) & (py_values < h)

    # Use only valid (px, py) pairs
    px_values = px_values[valid_indices]
    py_values = py_values[valid_indices]
    r_values = r_values[valid_indices]

    # Check the pixel intensities in the image at the (px, py) positions
    pixel_values = img[py_values, px_values]

    # Find the first radius where the pixel value exceeds the threshold
    over_threshold_indices = np.where(pixel_values > threshold)[0]
    
    if len(over_threshold_indices) > 0:
        return r_values[over_threshold_indices[0]]



def find_outer_edge(img, center, angle, start_radius):
    h, w = img.shape
    max_r = min(h, w) // 2
    x, y = center

    # Create a range of radii starting from start_radius
    r_values = np.arange(start_radius, max_r)

    # Calculate all (px, py) positions for the given range of radii
    px_values = (x + r_values * np.cos(angle)).astype(int)
    py_values = (y + r_values * np.sin(angle)).astype(int)

    # Filter out (px, py) values that are out of bounds
    valid_indices = (px_values >= 0) & (px_values < w) & (py_values >= 0) & (py_values < h)
    
    # Use only valid (px, py) pairs
    px_values = px_values[valid_indices]
    py_values = py_values[valid_indices]
    r_values = r_values[valid_indices]

    # Create the profile by extracting pixel intensities for valid (px, py) pairs
    profile = img[py_values, px_values]

    # Now we perform the same edge detection logic, but vectorized
    # Find positions where intensity drops by 40% (0.6 factor) over 5 pixels
    profile_shifted = profile[5:]  # Profile shifted by 5 pixels
    drops = profile[:-5] * 0.6 > profile_shifted

    # Ensure intensity never goes up again after the drop
    for i in np.where(drops)[0]:
        if np.all(profile[i+5:] <= profile[i+5]):
            return start_radius + i

    return max_r  # Return max_r if no suitable edge is found

def find_outer_edge1(img, center, angle, start_radius):
    h, w = img.shape
    max_r = min(h, w) // 2
    x, y = center
    profile = []
    for r in range(start_radius, max_r):
        px = int(x + r * np.cos(angle))
        py = int(y + r * np.sin(angle))
        if px < 0 or px >= w or py < 0 or py >= h:
            break
        profile.append(img[py, px])
    
    for i in range(len(profile) - 5):
        if profile[i] * 0.6 > profile[i+5]:
            # Check if intensity never goes up again
            if all(profile[j] <= profile[i+5] for j in range(i+5, len(profile))):
                return start_radius + i
    
    return max_r  # Return max_r if no suitable edge is found

def fit_circle(x, y, outlier_percentage=0.2):
    def calc_R(xc, yc):
        """ Calculate the distance of each 2D point from the center (xc, yc) """
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        """ Calculate the algebraic distance between the data points and the mean circle """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Add error handling for degenerate cases (e.g., too few points)
    if len(x) < 3:
        return None, None, None, None

    # Initial estimate of the circle's center
    center_estimate = np.mean(x), np.mean(y)
    
    # Use least-squares optimization to find the center
    center, _ = optimize.leastsq(f_2, center_estimate)
    
    # Calculate the distances from the center
    xc, yc = center
    Ri = calc_R(xc, yc)
    
    # Determine the number of points to discard as outliers
    num_points = len(Ri)
    num_outliers = int(outlier_percentage * num_points / 2)  # 10% from each end

    # Sort the distances and remove the outliers from both ends
    sorted_indices = np.argsort(Ri)
    inlier_indices = sorted_indices[num_outliers:-num_outliers]  # Exclude top and bottom 10%

    # Recalculate the center using only inliers
    x_inliers = x[inlier_indices]
    y_inliers = y[inlier_indices]

    # Re-fit the circle using the inliers only
    def f_2_inliers(c):
        """ Recalculate using inliers only """
        Ri_inliers = np.sqrt((x_inliers - c[0])**2 + (y_inliers - c[1])**2)
        return Ri_inliers - Ri_inliers.mean()

    center, _ = optimize.leastsq(f_2_inliers, center_estimate)
    
    # Final center and radius
    xc, yc = center
    Ri_inliers = np.sqrt((x_inliers - xc)**2 + (y_inliers - yc)**2)
    R = Ri_inliers.mean()
    residu = np.sum((Ri_inliers - R)**2)
    
    return xc, yc, R, residu


def fit_circle0(x, y):
    def calc_R(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.mean(y)
    center, _ = optimize.leastsq(f_2, center_estimate)
    
    xc, yc = center
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    residu = np.sum((Ri - R)**2)
    
    return xc, yc, R, residu

def analyze_donut_edges(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply 5x5 Gaussian blur
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Initial center estimate using centroid
    M = cv2.moments(img_blur)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    center = (center_x, center_y)
    
    # Generate angles for radial scanning
    num_angles = 360
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # Threshold for inner edge detection
    threshold = np.mean(img_blur)*1.3
    
    # Calculate distances to inner and outer edges
    inner_distances = [find_inner_edge(img_blur, center, angle, threshold) for angle in angles]
    outer_distances = [find_outer_edge(img_blur, center, angle, int(np.mean(inner_distances) * 1.2)) for angle in angles]
    
    # Convert polar coordinates to Cartesian for circle fitting
    inner_x = center[0] + inner_distances * np.cos(angles)
    inner_y = center[1] + inner_distances * np.sin(angles)
    outer_x = center[0] + outer_distances * np.cos(angles)
    outer_y = center[1] + outer_distances * np.sin(angles)
    
    # Fit circles to inner and outer edge points
    inner_xc, inner_yc, inner_R, inner_residu = fit_circle(inner_x, inner_y)
    outer_xc, outer_yc, outer_R, outer_residu = fit_circle(outer_x, outer_y)
    
    # Visualization
    plt.figure(figsize=(20, 10))
    
    # Original image with detected edges and best-fit circles
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.plot(center[0], center[1], 'r+', markersize=15, label='Initial Center')
    plt.plot(inner_xc, inner_yc, 'g+', markersize=15, label='Inner Center')
    plt.plot(outer_xc, outer_yc, 'b+', markersize=15, label='Outer Center')
    inner_circle = plt.Circle((inner_xc, inner_yc), inner_R, color='r', fill=False)
    outer_circle = plt.Circle((outer_xc, outer_yc), outer_R, color='b', fill=False)
    plt.gca().add_artist(inner_circle)
    plt.gca().add_artist(outer_circle)
    plt.title('Image with Best-Fit Circles')
    plt.legend()
    plt.axis('off')
    
    # Plot of edge distances vs angle
    plt.subplot(122, projection='polar')
    plt.polar(angles, inner_distances, 'r.', label='Inner Edge Points')
    plt.polar(angles, outer_distances, 'b.', label='Outer Edge Points')
    plt.polar(angles, [inner_R]*len(angles), 'r-', label='Inner Best-Fit')
    plt.polar(angles, [outer_R]*len(angles), 'b-', label='Outer Best-Fit')
    plt.title('Edge Distances vs Angle')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return (inner_xc, inner_yc, inner_R), (outer_xc, outer_yc, outer_R)

# Example usage
image_path = 'col.jpg'
inner_circle, outer_circle = analyze_donut_edges(image_path)

print(f"Inner circle: Center = ({inner_circle[0]:.2f}, {inner_circle[1]:.2f}), Diameter = {inner_circle[2]*2:.2f} pixels")
print(f"Outer circle: Center = ({outer_circle[0]:.2f}, {outer_circle[1]:.2f}), Diameter = {outer_circle[2]*2:.2f} pixels")
