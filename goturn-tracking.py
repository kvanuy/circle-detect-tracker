import cv2, os
from ultralytics import YOLO
import torch


CURRENT_FILE = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':

    tracker = cv2.TrackerGOTURN_create()
    video_path = os.path.join(CURRENT_FILE, 'vid1.mp4')

    video = cv2.VideoCapture(video_path)

    n = 0
    track_dir = os.path.join(CURRENT_FILE, 'track_results', f'vid_{n}')
    while os.path.exists(track_dir) is True:
        n+=1
        track_dir = os.path.join(CURRENT_FILE, 'track_results', f'vid_{n}')
    os.makedirs(track_dir)
    
    if not video.isOpened():
        print('count not open video')
        exit()
        
    ok,frame = video.read()
    if not ok:
        print("cannot read video file")
        exit()


    #set up YOLO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = os.path.join(CURRENT_FILE, 'train8', 'weights', 'best.pt')
    trained_model = YOLO(weight_path)

    num = 0

    while ok:
        #run yolo for obj detection    
        timer = cv2.getTickCount()

        #store the video img as an img
        raw_img_path = os.path.join(track_dir, f'rawimg_{num}.jpg')
        cv2.imwrite(raw_img_path, frame)
        obj_results = trained_model(raw_img_path)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        #parse the result to access the ROI coords
        
        for i in obj_results:
            obj = i.boxes
            for box_num in range(0, len(obj.xyxy)):
                #access the bounding box coord
                x1,y1,x2,y2 = obj.xyxy[box_num]
                x1,y1,x2,y2 = int(x1.item()),int(y1.item()), int(x2.item()), int(y2.item())


                x1,y1,w,h = obj.xywh[box_num]
                x1,y1,w,h = int(x1.item()),int(y1.item()), int(w.item()), int(h.item())
                x1, y1 = x1-w//2, y1-h//2

                #cv2.rectangle( frame, (x1,y1), (x1+w, y1+h), (0, 0, 255), 2, 1)



                if num == 0:
                    ok = tracker.init(frame, (x1,y1,w,h))

                    
        else:
            ok, bbox = tracker.update(frame)
            #print(bbox)
            #track success
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle( frame, p1, p2, (250, 0, 0), 2, 1)
            else:
                cv2.putText(frame,'Tracking Failure', (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
            cv2.putText(frame, "GOTURN Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        
        file_nm = os.path.join(CURRENT_FILE, 'track_results', f'vid_{n}', f'track-frame{num}.jpg')
        cv2.imwrite(file_nm, frame)

        num += 1
        ok,frame = video.read()

                