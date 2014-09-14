import Image
import ImageDraw
import ImageFont
import cv2

image = Image.new("RGBA", (310,90), (255,255,255))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("resources/MANDATORY.ttf", 20)

draw.text((10, 10), "0 1 2 3 4 5 6 7 8 9 A B C D E F", (0,0,0), font=font)
draw.text((10, 30), "G H I J K L M N O P Q R S T U V", (0,0,0), font=font)
draw.text((10, 50), "W X Y Z", (0,0,0), font=font)
#img_resized = image.resize((10,10), Image.ANTIALIAS)
image.save("resources/font.png")
