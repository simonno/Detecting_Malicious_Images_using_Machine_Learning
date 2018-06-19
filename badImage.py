import cv2  # OpenCV 2


def to_bit_generator(msg):
    """Converts a message into a generator which returns 1 bit of the message
    each time."""
    for c in msg:
        o = c
        for i in range(8):
            yield (o & (1 << i)) >> i


def create_bad_image(path_image, path_malware):
    # Create a generator for the hidden message
    hidden_message = to_bit_generator(open(path_malware, "rb").read() * 10)

    # Read the original image
    img = cv2.imread(path_image, cv2.IMREAD_COLOR)
    width, height, rgb = img.shape
    for h in range(width):
        for w in range(height):
            for i in range(rgb):
                # Write the hidden message into the 3th bit
                bit = 251 + next(hidden_message) * 4
                img[h][w][i] = img[h][w][i] & bit
    # Write out the image with hidden message
    cv2.imwrite(path_image, img)
