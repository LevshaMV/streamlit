import streamlit as st
import cv2
import subprocess
from tempfile import NamedTemporaryFile
import os
import statistics
import plotly.express as px
from ultralytics import YOLO
import random

def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.read())
        
def weight_calculation(x1,y1, x2, y2):
    pixel2cm = 0.03
    pi = 3.14
    x = abs(x2-x1)
    y = abs(y2-y1)

    x *= pixel2cm
    y *= pixel2cm
    res = int(f'{pi * x ** 3/ 5:.0f}')
    return 120 if res > 1000 else res

def main():
    st.title("Video Processing with OpenCV and Streamlit")

    video_data = st.file_uploader("Upload video file", ['mp4', 'mov', 'avi'])

    if video_data:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_to_save, NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_result:
            temp_filename_to_save = temp_file_to_save.name
            temp_filename_result = temp_file_result.name
            model = YOLO('runs/detect/train2/weights/best.pt')
            model.fuse()

            # Сохраняем загруженное видео в временный файл
            write_bytesio_to_file(temp_filename_to_save, video_data)

            # Открываем видео с помощью OpenCV
            cap = cv2.VideoCapture(temp_filename_to_save)
            if not cap.isOpened():
                st.error("Ошибка при открытии видеофайла")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            out_mp4 = cv2.VideoWriter(temp_filename_result, fourcc_mp4, frame_fps, (width, height), isColor=False)
            
            weights = {}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=640, verbose=False, tracker='botsort.yaml')
                if results[0].boxes.id != None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)

                    for box,id in zip(boxes,ids):
                        random.seed(int(id))
                        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                        weight = weight_calculation(box[0], box[1], box[2], box[3])
                        if id not in weights:
                            weights[id] = weight
                        else:
                            weights[id] = max(weights[id],weight)
                            #weights[id] = (weights[id]+weight)/2

                        cv2.rectangle(frame, (box[0], box[1]), (box[2],box[3],), color, 2)
                        cv2.putText(
                            frame,
                            f"id {id}, weight: ~{weights[id]:.0f}g",
                            (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2,
                        )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #gray
                out = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                out_mp4.write(out)

            cap.release()
            out_mp4.release()

            if not os.path.exists(temp_filename_result):
                st.error("Ошибка: результат обработки видео с помощью OpenCV не был сохранен")
                return

            # Конвертируем видео в формат H264 с помощью ffmpeg
            converted_video = temp_filename_result + '_converted.mp4'
            result = subprocess.run(f"ffmpeg -y -i {temp_filename_result} -c:v libx264 {converted_video}", shell=True)

            if result.returncode != 0:
                st.error("Ошибка при конвертации видео с помощью ffmpeg")
                return

            if not os.path.exists(converted_video):
                st.error(f"Ошибка: файл {converted_video} не был создан")
                return

            # Показываем результаты
            st.header("Original Video")
            st.video(temp_filename_to_save)

            #st.header("Output from OpenCV (MPEG-4)")
            #st.video(temp_filename_result)

            st.header("After conversion to H264")
            st.video(converted_video)
            
            st.header("Statistics from the video")
            data = weights
            df = list(data)
            fig_hist = px.histogram(df, nbins=20, title="Результат обработки видео",labels={"variable": "Полученные данные"},color_discrete_sequence=['#00A86B']).update_xaxes(categoryorder='total ascending')
            fig_hist.update_layout(
                xaxis_title="Вес томатов, г.",
                yaxis_title="Количество томатов, шт.",
                bargap=0.1,
                showlegend = False,
                legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.8,
                itemwidth=100),
                )

            fig_hist.update_layout(plot_bgcolor="#e2eeee")

            fig_hist.add_annotation(text=f'Полученные данные:',
                                showarrow=False,
                                xref='paper',
                                yref='paper',
                                height = 20,
                                width = 300,
                                x=0.95,
                                y=0.97,
                                bgcolor = 'white',
                                font=dict(
                                    weight=500,
                                    size=14,),
                                )
            fig_hist.add_annotation(text=f'\t\t\t\tОбщее количество томатов, шт: {len(df)} <br>\t\t\t\tВес, кг:  {sum(df)//1000}<br>\t\t\t\tСреднее значение, г: {sum(df)//len(df)} <br>\t\t\t\tМода, г: {statistics.mode(data)}<br>\t\t\t\tМаксимальное значение, г: {max(df)}<br>\t\t\t\tМинимальное значение, г: {min(df)}', 
                                align='left',
                                showarrow=False,
                                xref='paper',
                                yref='paper',
                                height = 120,
                                width = 300,
                                x=0.95,
                                y=0.93,
                                bgcolor = 'white',)


            newnames = {'0':'Томаты',}
            fig_hist.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                                legendgroup = newnames[t.name],
                                                hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                                )
                            )
            st.plotly_chart(fig_hist)

if __name__ == "__main__":
    main()