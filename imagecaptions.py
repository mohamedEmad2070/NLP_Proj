from transformers import BlipProcessor, BlipForConditionalGeneration
from tkinter import Tk, filedialog, Label, Button
from PIL import Image, ImageTk
from deep_translator import GoogleTranslator

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize the translator
translator = GoogleTranslator(source='en', target='ar')

# Function to generate captions
def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Modify the upload_image function to include Arabic translation
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        caption = generate_caption(file_path)
        translated_caption = translator.translate(caption)
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.configure(image=img_tk)
        image_label.image = img_tk
        caption_label.config(text=f"Caption: {caption}\nTranslated Caption: {translated_caption}")

# GUI Setup
root = Tk()
root.title("Image Caption Generator")
root.geometry("400x500")
root.configure(bg="#2c3e50")  # Set background color

Label(root, text="Upload an Image to Generate a Caption", font=("Arial", 14), bg="#2c3e50", fg="#ecf0f1").pack(pady=10)  # Set text and background colors

image_label = Label(root, bg="#34495e")  # Set background color for image area
image_label.pack(pady=10)

Button(root, text="Upload Image", command=upload_image, bg="#1abc9c", fg="#ecf0f1", font=("Arial", 12), relief="flat", padx=10, pady=5).pack(pady=10)  # Modern button style

# Update the font of the caption_label to support Arabic characters
caption_label = Label(root, text="", wraplength=350, font=("Arial", 12), bg="#2c3e50", fg="#ecf0f1")  # Use a font that supports Arabic
caption_label.pack(pady=20)

root.mainloop()