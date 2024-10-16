import time
import numpy
import numpy as np
import matplotlib as plt
from PIL import Image
import math

def rational_interpolation_bulirsch_stoer(x, y, x_new):
    n = len(x) - 1
    m = len(y) - 1

    diff_table = np.zeros((n + 1, m + 1))
    diff_table[:, 0] = y

    for j in range(1, m + 1):
        for i in range(n - j + 1):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x[i + j] - x[i])

    # Evaluate the interpolated function at the new point
    y_new = 0.0
    x_product = np.ones_like(x_new)

    for j in range(m + 1):
        y_new += diff_table[0, j] * x_product
        x_product *= (x_new - x[j])

    return y_new

def barycentric_interpolation(x, y, xi):
    n = len(x)
    w = np.ones(n)

    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] *= (xi - x[k]) / (x[j] - x[k])

    numerator = np.sum(w * y)
    denominator = np.sum(w)
    if denominator == 0:
        result = 0
    else:
        result = numerator / denominator

    return result

def tester_function1(x):
    return np.sin(2 * np.pi * x) / (1 + 5 * x**2)

def test_interpolation_methods_fun1():
    # Define the interval and sample points
    x = np.linspace(-1, 1, 11)
    y = tester_function1(x)

    # Generate points for plotting
    x_plot = np.linspace(-1, 1, 100)
    y_true = tester_function1(x_plot)

    # Evaluate using barycentric interpolation
    y_barycentric = [barycentric_interpolation(x, y, xi) for xi in x_plot]

    # Evaluate using Bulirsch-Stoer interpolation
    y_bulirsch_stoer = [rational_interpolation_bulirsch_stoer(x, y, xi) for xi in x_plot]

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_true, label='True Function')
    plt.plot(x_plot, y_barycentric, label='Barycentric Interpolation')
    plt.plot(x_plot, y_bulirsch_stoer, label='Bulirsch-Stoer Interpolation')
    plt.scatter(x, y, color='red', label='Sample Points')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Rational Interpolation Methods')
    plt.show()

def compare_interpolation_methods_fun1():
    # Define the interval and sample points
    x = np.linspace(0, 2, 11)
    y = tester_function1(x)

    # Generate points for evaluation
    x_eval = np.linspace(0, 2, 100)
    y_true = tester_function1(x_eval)

    # Evaluate using barycentric interpolation
    start_time = time.time()
    y_barycentric = [barycentric_interpolation(x, y, xi) for xi in x_eval]
    barycentric_time = time.time() - start_time

    # Evaluate using Bulirsch-Stoer interpolation
    start_time = time.time()
    y_bulirsch_stoer = [rational_interpolation_bulirsch_stoer(x, y, xi) for xi in x_eval]
    bulirsch_stoer_time = time.time() - start_time

    # Calculate interpolation errors
    barycentric_error = np.abs(y_barycentric - y_true)
    bulirsch_stoer_error = np.abs(y_bulirsch_stoer - y_true)

    # Print the results
    print("Interpolation Method Comparison:")
    print("-------------------------------")
    print(f"Barycentric Interpolation Error: {np.mean(barycentric_error):.6f}")
    print(f"Bulirsch-Stoer Interpolation Error: {np.mean(bulirsch_stoer_error):.6f}")
    print(f"Barycentric Interpolation Time: {barycentric_time:.6f} seconds")
    print(f"Bulirsch-Stoer Interpolation Time: {bulirsch_stoer_time:.6f} seconds")

def tester_function2(x):
    return np.exp(-x) * np.cos(2 * np.pi * x)

def test_interpolation_methods_fun2():
    # Define the interval and sample points
    x = np.linspace(-1, 1, 11)
    y = tester_function2(x)

    # Generate points for plotting
    x_plot = np.linspace(-1, 1, 100)
    y_true = tester_function2(x_plot)

    # Evaluate using barycentric interpolation
    y_barycentric = [barycentric_interpolation(x, y, xi) for xi in x_plot]

    # Evaluate using Bulirsch-Stoer interpolation
    y_bulirsch_stoer = [rational_interpolation_bulirsch_stoer(x, y, xi) for xi in x_plot]

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_true, label='True Function')
    plt.plot(x_plot, y_barycentric, label='Barycentric Interpolation')
    plt.plot(x_plot, y_bulirsch_stoer, label='Bulirsch-Stoer Interpolation')
    plt.scatter(x, y, color='red', label='Sample Points')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Rational Interpolation Methods')
    plt.show()

def compare_interpolation_methods_fun2():
    # Define the interval and sample points
    x = np.linspace(0, 2, 11)
    y = tester_function2(x)

    # Generate points for evaluation
    x_eval = np.linspace(0, 2, 100)
    y_true = tester_function2(x_eval)

    # Evaluate using barycentric interpolation
    start_time = time.time()
    y_barycentric = [barycentric_interpolation(x, y, xi) for xi in x_eval]
    barycentric_time = time.time() - start_time

    # Evaluate using Bulirsch-Stoer interpolation
    start_time = time.time()
    y_bulirsch_stoer = [rational_interpolation_bulirsch_stoer(x, y, xi) for xi in x_eval]
    bulirsch_stoer_time = time.time() - start_time

    # Calculate interpolation errors
    barycentric_error = np.abs(y_barycentric - y_true)
    bulirsch_stoer_error = np.abs(y_bulirsch_stoer - y_true)

    # Print the results
    print("Interpolation Method Comparison:")
    print("-------------------------------")
    print(f"Barycentric Interpolation Error: {np.mean(barycentric_error):.6f}")
    print(f"Bulirsch-Stoer Interpolation Error: {np.mean(bulirsch_stoer_error):.6f}")
    print(f"Barycentric Interpolation Time: {barycentric_time:.6f} seconds")
    print(f"Bulirsch-Stoer Interpolation Time: {bulirsch_stoer_time:.6f} seconds")

def zooming_barycentric(matrix, k):
    n = len(matrix)
    m = len(matrix[0])
    ans = []
    for i in range(k * n):
        cur = []
        for j in range(k * m):
            cur.append('inf')
        ans.append(cur)

    for i in range(n):
        for j in range(m):
            ans[i * k][j * k] = matrix[i][j]

    for i in range(k * n):
        if i % k == 0:
            x = []
            y = []
            for j in range(k * m):
                if ans[i][j] != 'inf':
                    x.append(j)
                    y.append(ans[i][j])

            for j in range(k * m):
                if ans[i][j] == 'inf':
                    ans[i][j] = barycentric_interpolation(x, y, j)

    for j in range(k * m):
        x = []
        y = []
        for i in range(k * n):
            if ans[i][j] != 'inf':
                x.append(i)
                y.append(ans[i][j])

        for i in range(k * n):
            if(ans[i][j] == 'inf'):
                ans[i][j] = barycentric_interpolation(x, y, i)


    return ans

def zooming_neighbor(matrix, k):
    n = len(matrix)
    m = len(matrix[0])
    ans = []
    for i in range(k * n):
        cur = []
        for j in range(k * m):
            cur.append('inf')
        ans.append(cur)

    for i in range(n):
        for j in range(m):
            ans[i * k][j * k] = matrix[i][j]

    for i in range(k * n):
        for j in range(k * m):
            cur_i = i - i % k
            cur_j = j - j % k
            ans[i][j] = ans[cur_i][cur_j]

    return ans

def convert_to_separate(image):
    n = image.shape[0]
    m = image.shape[1]
    red = []
    green = []
    blue = []
    for i in range(n):
        r = []
        g = []
        b = []
        for j in range(m):
            r.append(image[i][j][0])
            g.append(image[i][j][1])
            b.append(image[i][j][2])
        red.append(r)
        green.append(g)
        blue.append(b)
    return [red, green, blue]

def convert_back(red, green, blue):
    image = []
    n = len(red)
    m = len(red[0])
    for i in range(n):
        lst = []
        for j in range(m):
            lst.append([red[i][j], green[i][j], blue[i][j]])
        image.append(lst)
    return image

# to test zooming
def action(image_path, lower_x, lower_y, width, height, scalar):
    cropped_image = crop_image(image_path, lower_x, lower_y, width, height)
    # cropped_image.show()
    image = numpy.array(cropped_image)
    r, g, b = convert_to_separate(image)
    r = zooming_neighbor(r, scalar)
    g = zooming_neighbor(g, scalar)
    b = zooming_neighbor(b, scalar)
    im = numpy.array(convert_back(r, g, b))
    img = Image.fromarray(im.astype(np.uint8))
    img.show()

# last task
def bilinear_interpolation(image, x, y):
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1
    y2 = y1 + 1

    # Get the image dimensions
    width, height = image.size

    # Ensure the interpolated coordinates are within the image boundaries
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # Get the pixel values of the four surrounding pixels
    pixel1 = image.getpixel((x1, y1))
    pixel2 = image.getpixel((x2, y1))
    pixel3 = image.getpixel((x1, y2))
    pixel4 = image.getpixel((x2, y2))

    # Calculate the weights for interpolation
    weight1 = (x2 - x) * (y2 - y)
    weight2 = (x - x1) * (y2 - y)
    weight3 = (x2 - x) * (y - y1)
    weight4 = (x - x1) * (y - y1)

    # Perform interpolation for each channel (R, G, B)
    interpolated_pixel = (
        int(pixel1[0] * weight1 + pixel2[0] * weight2 + pixel3[0] * weight3 + pixel4[0] * weight4),
        int(pixel1[1] * weight1 + pixel2[1] * weight2 + pixel3[1] * weight3 + pixel4[1] * weight4),
        int(pixel1[2] * weight1 + pixel2[2] * weight2 + pixel3[2] * weight3 + pixel4[2] * weight4)
    )

    return interpolated_pixel

def rotate_image_with_bilinear_interpolation(path):
    image = Image.open(path)

    angle = float(input("Enter the rotation angle (in degrees): "))

    angle_rad = angle * (math.pi / 180)

    width, height = image.size

    new_width = int(abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad)))
    new_height = int(abs(height * math.cos(angle_rad)) + abs(width * math.sin(angle_rad)))

    rotated_image = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

    center_x = new_width // 2
    center_y = new_height // 2

    for x in range(new_width):
        for y in range(new_height):
            original_x = (x - center_x) * math.cos(angle_rad) + (y - center_y) * math.sin(angle_rad) + width // 2
            original_y = -(x - center_x) * math.sin(angle_rad) + (y - center_y) * math.cos(angle_rad) + height // 2

            # Perform bilinear interpolation to get the pixel value
            pixel = bilinear_interpolation(image, original_x, original_y)

            rotated_image.putpixel((x, y), pixel)

    rotated_image.show()

def crop_image(image, lower_x, lower_y, width, height):
    img = Image.open(image)

    upper_x = lower_x + width
    upper_y = lower_y + height
    cropped_img = img.crop((lower_x, lower_y, upper_x, upper_y))

    return cropped_img