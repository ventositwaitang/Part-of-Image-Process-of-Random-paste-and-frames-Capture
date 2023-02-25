import cv2

#==================== Setting up the file =================
videoFile = "C:/Users/User/Desktop/JIG/Project/videoCapture_every_nSeconds/gg.MP4"
vidcap = cv2.VideoCapture(videoFile)
success, frame = vidcap.read()

#==================== Setting up parameters ====================
count = 0
# Capture every n seconds (here, n = 5)
seconds_skips = 5
# fps = vidcap.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
# multiplier = fps * seconds_skips

# # set size of captured video
# vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

#==================== Initiate Process ====================

while vidcap.isOpened():
    # frameId = int(
    #     round(vidcap.get(1))
    # )  # current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    num = count * seconds_skips
    cv2.imwrite(
        "C:/Users/User/Desktop/JIG/Project/videoCapture_every_nSeconds/%d second.jpg"
        % num,
        frame,
    )
    count += 1
    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * float(1000 * seconds_skips)))
    success, frame = vidcap.read()
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

# release all sources
vidcap.release()
print("Complete")

"""""" """""" """""
import cv2

vid = cv2.VideoCapture('C:/Users/User/Desktop/JIG/Project/snapshot frame every N second/IMG_8905.MOV')
framerate = cap.get(5)
x=1

if not os.path.exists('frames'):
    os.makedirs('frames')

index = 0
while(True):
    # Capture frame-by-frame
    ret, frame = vid.read()
    if not ret:
        break
    # comment this
    # cap.release()
    # Our operations on the frame come here
    name = 'C:/Users/User/Desktop/JIG/Project/snapshot frame every N second/IMG_8905.MOV' + str(index) + '.jpg'
    if index%2==0:
        time.sleep(2)
        cv2.imwrite(name, frame)
    index += 1


def imgcap():
    cap = cv2.VideoCapture(0)
    framerate = cap.get(5)
    x=1

    while(True):

        ret, frame = cap.read()

        filename = 'Capture' +  str(int(x)) + ".png"
            x=x+1
            cv2.imwrite(filename, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # else:
        #     print("Ret False")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
imgcap()
""" """""" """""" ""

