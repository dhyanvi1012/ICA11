from src.predict import predict_image

import logging
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

# Call the predict_image function with your image data
# Replace 'image_path' with the path to your image file
image_path = r"cat1.jpg"
predicted_class = predict_image(image_path)

print(f"Predicted class: {predicted_class}")
logging.info(f'Predicted class: {predicted_class}')