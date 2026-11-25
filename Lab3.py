
import numpy as np

print("NumPy version:", np.__version__)

# =====================================================================
# Task 1: Creating arrays (2 marks)
# =====================================================================
print("\n=== Task 1 ===")
# 1a. Make an array `temps` with this week's temperatures:
#     31, 28, 35, 30, 33, 29, 36
#     Print it, then print its dtype, shape and size.

# Storing the temperatures of all seven days in an array
temps = np.array([31, 28, 35, 30, 33, 29, 36])
print("Temperatures:", temps)
print("Data type:", temps.dtype)
print("Shape:", temps.shape)
print("Size:", temps.size)

# 1b. Make `months`: the numbers 1 to 12 as a 3 x 4 table
#     (use np.arange and reshape). Print it and its shape.

# Creating numbers from 1 to 12 and arranging them in 3 rows and 4 columns
months = np.arange(1, 13).reshape(3, 4)
print("Months:\n", months)
print("Months shape:", months.shape)

# 1c. Print an array of five zeros, and a 2 x 3 array of ones.

# Creating arrays filled with zeros and ones
print("Zeros:", np.zeros(5))
print("Ones:\n", np.ones((2, 3)))

# 1d. Make a random generator with seed 42 and roll 10 dice
#     (whole numbers from 1 to 6). Print the result.

# Using seed 42 so the dice results stay the same each time
rng = np.random.default_rng(42)
dice = rng.integers(1, 7, size=10)
print("Dice rolls:", dice)


# =====================================================================
# Task 2: Indexing and masks (2.5 marks)  (use `temps` from Task 1)
# =====================================================================
print("\n=== Task 2 ===")
# 2a. Print the first and the last temperature.

# Accessing the first and last elements using their indexes
print("First temperature:", temps[0])
print("Last temperature:", temps[-1])

# 2b. Print the weekend temperatures (the last two days).

# The last two values represent the weekend
print("Weekend temperatures:", temps[-2:])

# 2c. Print the temperatures in reverse order.

# Using ::-1 to reverse the array
print("Reverse temperatures:", temps[::-1])

# 2d. Print only the hot days (above 32), how many there are,
#     and their average.  No loops!

# Filtering temperatures greater than 32 using a boolean mask
hot_days = temps[temps > 32]
print("Hot temperatures:", hot_days)
print("Number of hot days:", hot_days.size)
print("Average of hot days:", hot_days.mean())

# 2e. Print the temperatures from 29 to 33 (both included). Hint: &

# Checking both conditions together
selected_temps = temps[(temps >= 29) & (temps <= 33)]
print("Temperatures between 29 and 33:", selected_temps)

# 2f. Use np.where to label each day "Hot" (33 or more) or "OK".

# Assigning labels depending on the temperature
labels = np.where(temps >= 33, "Hot", "OK")
print("Day labels:", labels)


# =====================================================================
# Task 3: Gradebook (3 marks)
# Rows = students, columns = 4 subjects.
# =====================================================================
print("\n=== Task 3 ===")
names = np.array(["Ali", "Sara", "Usman", "Hina", "Bilal"])
marks = np.array([[78, 85, 62, 90],
                  [92, 88, 95, 79],
                  [45, 60, 48, 70],
                  [88, 94, 91, 85],
                  [65, 42, 75, 58]])
# 3a. Print the shape of marks.

# Displaying the number of students and subjects
print("Marks shape:", marks.shape)

# 3b. Print Usman's marks (one row) and all marks of subject 3
#     (column index 2).

# Usman is at row index 2, and subject 3 is at column index 2
print("Usman's marks:", marks[2])
print("Subject 3 marks:", marks[:, 2])

# 3c. Print the average of each subject and the average of each student.

# Axis 0 calculates subject averages, while axis 1 calculates student averages
print("Average of each subject:", marks.mean(axis=0))
print("Average of each student:", marks.mean(axis=1))

# 3d. Print the NAME of the student with the highest average.
#     Hint: argmax gives a position; use it on `names`.

# Finding the position of the highest student average
student_averages = marks.mean(axis=1)
top_student = np.argmax(student_averages)
print("Student with highest average:", names[top_student])

# 3e. Subject weights are 0.3, 0.3, 0.2, 0.2. Print each student's
#     weighted total, rounded to 1 decimal. (Broadcasting!)

# Multiplying each subject's marks by its weight
weights = np.array([0.3, 0.3, 0.2, 0.2])
weighted_totals = np.sum(marks * weights, axis=1)
print("Weighted totals:", np.round(weighted_totals, 1))

# 3f. Add 5 grace marks to everyone, but no mark may go above 100.
#     Print the new table. Hint: np.clip

# Adding 5 marks and keeping all values within the 0 to 100 range
new_marks = np.clip(marks + 5, 0, 100)
print("Marks after grace marks:\n", new_marks)

# 3g. Print how many students failed (below 50) in each subject.

# Counting the marks below 50 column by column
failed_students = np.sum(marks < 50, axis=0)
print("Failed students in each subject:", failed_students)


# =====================================================================
# Task 4: An image is an array (2.5 marks)
# A 5 x 5 grayscale image: 0 = black, 255 = white.
# =====================================================================
print("\n=== Task 4 ===")
img = np.array([[ 10,  10, 200,  10,  10],
                [ 10,  10, 220,  10,  10],
                [210, 230, 250, 230, 210],
                [ 10,  10, 220,  10,  10],
                [ 10,  10, 200,  10,  10]])
# 4a. Print the shape, the darkest value and the brightest value.

# Finding the image dimensions and its minimum and maximum pixel values
print("Image shape:", img.shape)
print("Darkest value:", img.min())
print("Brightest value:", img.max())

# 4b. Print the inverted image (white becomes black). Hint: 255 - ...

# Subtracting every pixel value from 255 to invert the image
inverted_img = 255 - img
print("Inverted image:\n", inverted_img)

# 4c. Print a black-and-white version: 1 where the pixel is above 128,
#     else 0. Hint: a mask, then .astype(int)

# Pixels above 128 become 1, and the remaining pixels become 0
bw_img = (img > 128).astype(int)
print("Black and white image:\n", bw_img)

# 4d. Print row 2 normalized to the range 0 to 1, rounded to 2 decimals.

# Taking row index 2 and dividing its values by 255
normalized_row = img[2] / 255
print("Normalized row 2:", np.round(normalized_row, 2))

# 4e. Print the 3 x 3 centre of the image (crop).

# Selecting the middle three rows and middle three columns
center_img = img[1:4, 1:4]
print("Centre of image:\n", center_img)
