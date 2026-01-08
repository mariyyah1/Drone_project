
# import asyncio
# import math
# import cv2
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

# from mavsdk import System
# from mavsdk.mission import MissionItem, MissionPlan
# from mavsdk.action import OrbitYawBehavior

# # ---------------- CONFIG ----------------
# VIDEO_PATH = "vid.webm"        
# AREA_THRESHOLD = 3000          
# SPIRAL_RADII = range(5, 20, 5)
# CAMERA_TOPIC = '/world/default/model/x500_gimbal_0/link/camera_link/sensor/camera/image'
# # ----------------------------------------

# object_detected = False
# cv_enabled = False  
# markers = []      

# # -------- SIMPLE CV --------
# def simple_cv_detect(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_red = np.array([0, 120, 70])
#     upper_red = np.array([10, 255, 255])
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     area = cv2.countNonZero(mask)
#     return area > AREA_THRESHOLD


# async def cv_demo_monitor(drone: System):
#     global object_detected, cv_enabled

#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if not cap.isOpened():
#         print("❌ Demo video not found")
#         return

#     print("👀 CV demo ready (waiting for search)")

#     while True:
#         if not cv_enabled:
#             await asyncio.sleep(0.2)
#             continue

#         ret, frame = cap.read()
#         if not ret:
#             break

#         if simple_cv_detect(frame):
#             print("🎯 Object detected (Simple CV)")
#             object_detected = True
#             await save_detection_gps(drone)
#             break

#         await asyncio.sleep(0.05)


# # -------- DRONE HELPERS --------
# async def landed(drone):
#     async for state in drone.telemetry.landed_state():
#         if state == state.ON_GROUND:
#             print("🛬 Drone landed")
#             break


# async def wait_one_orbit(drone: System, center_lat: float, center_lon: float, radius_m: float,
#                          circle_tolerance_m: float = 1.0,
#                          start_tolerance_deg: float = 10 * 1e-6):
#     """
#     تنتظر الدرون لإكمال دورة كاملة حول مركز معين
#     """
#     def distance_m(lat1, lon1, lat2, lon2):
#         R = 6371000
#         phi1 = math.radians(lat1)
#         phi2 = math.radians(lat2)
#         dphi = math.radians(lat2 - lat1)
#         dlambda = math.radians(lon2 - lon1)
#         a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
#         c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#         return R * c

#     # بداية الدورة
#     print("Waiting for drone to reach orbit circle...")
#     async for position in drone.telemetry.position():
#         d = distance_m(center_lat, center_lon, position.latitude_deg, position.longitude_deg)
#         if abs(d - radius_m) <= circle_tolerance_m:
#             start_lat = position.latitude_deg
#             start_lon = position.longitude_deg
#             print(f"Orbit started near: {start_lat}, {start_lon}")
#             break

#     # الانتظار حتى يغادر النقطة
#     async for position in drone.telemetry.position():
#         if (abs(position.latitude_deg - start_lat) > start_tolerance_deg or
#             abs(position.longitude_deg - start_lon) > start_tolerance_deg):
#             print("Left start point, orbit in progress...")
#             break

#     # الانتظار حتى العودة للنقطة نفسها (دورة كاملة)
#     print("Waiting for drone to come back to start of orbit (one full round)...")
#     async for position in drone.telemetry.position():
#         if (abs(position.latitude_deg - start_lat) <= start_tolerance_deg and
#             abs(position.longitude_deg - start_lon) <= start_tolerance_deg):
#             print("One full orbit completed!")
#             break


# async def save_detection_gps(drone):
#     pos = await anext(drone.telemetry.position())
#     lat = pos.latitude_deg
#     lon = pos.longitude_deg
#     alt = pos.absolute_altitude_m
#     print(f"📍 GPS SAVED → LAT:{lat}, LON:{lon}, ALT:{alt}")
#     markers.append((lat, lon, alt))
#     await add_marker_to_mission(drone, lat, lon, alt)


# async def add_marker_to_mission(drone, lat, lon, alt):
#     """
#     إضافة Marker مستقل على الخريطة (يمكن تعديله حسب GCS)
#     """
#     # هنا Marker مستقل، بعض الـ GCS قد تحتاج API خاص لإضافته فعليًا
#     print(f"📌 Marker added at LAT:{lat}, LON:{lon}, ALT:{alt}")


# # -------- SPIRAL SEARCH --------
# async def spiral_search(drone, lat, lon, alt):
#     global object_detected, cv_enabled

#     print("🔍 Spiral search started")

#     for radius in SPIRAL_RADII:
#         print(f"🔄 Starting orbit radius {radius} m")
#         await drone.action.do_orbit(
#             radius_m=radius,
#             velocity_ms=2.0,
#             yaw_behavior=OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER,
#             latitude_deg=lat,
#             longitude_deg=lon,
#             absolute_altitude_m=alt
#         )

#         await wait_one_orbit(drone, lat, lon, radius)
#         cv_enabled = True 

#         if object_detected:
#             print("🛑 Search stopped (CV triggered)")
#             break

#     print("✅ Spiral search completed")


# # -------- MAIN --------
# async def main():
#     global object_detected

#     drone = System()
#     print("🔌 Connecting...")
#     await drone.connect(system_address="udp://:14540")

#     async for state in drone.core.connection_state():
#         if state.is_connected:
#             print("✅ Drone connected")
#             break

#     print("🔓 Arming")
#     await drone.action.arm()
#     await asyncio.sleep(2)

#     print("🚀 Takeoff")
#     await drone.action.takeoff()
#     await asyncio.sleep(5)

#     home = await anext(drone.telemetry.home())
#     home_lat = home.latitude_deg
#     home_lon = home.longitude_deg
#     home_alt = home.absolute_altitude_m + 2

#     cv_task = asyncio.create_task(cv_demo_monitor(drone))

# # 0.000135 هي القيمة التي تحول الـ 15 متر إلى إحداثيات جغرافية
#     target_lat = home_lat + (11.11 * 0.000009)  # Y = 11.11
#     target_lon = home_lon + (15.00 * 0.000009)  # X = 15.00

#     mission = MissionPlan([
#         MissionItem(
#             target_lat, target_lon, 2,
#             3, True,
#             float("nan"), float("nan"),
#             MissionItem.CameraAction.NONE,
#             float("nan"), float("nan"),
#             1, 8, float("nan"),
#             MissionItem.VehicleAction.NONE
#         )
#     ])

#     print("📤 Uploading mission")
#     await drone.mission.upload_mission(mission)

#     print("▶️ Starting mission")
#     await drone.mission.start_mission()

#     async for progress in drone.mission.mission_progress():
#         print(f"Mission: {progress.current}/{progress.total}")
#         if progress.current == progress.total:
#             print("🏁 Mission done")
#             break

#     await drone.mission.pause_mission()

#     # Spiral search
#     await spiral_search(drone, target_lat, target_lon, home_alt)

#     # RTL if detected
#     if object_detected:
#         print("🏠 RTL (CV Triggered)")
#         await drone.action.return_to_launch()

#     await landed(drone)
#     cv_task.cancel()

#     print("✅ Demo finished")


# asyncio.run(main())





# ////////////////////////////////////////////////////////////////////////////////




# import asyncio
# import math
# import cv2
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from ultralytics import YOLO

# from mavsdk import System
# from mavsdk.mission import MissionItem, MissionPlan
# from mavsdk.action import OrbitYawBehavior

# # ---------------- CONFIG ----------------
# # المسار الخاص بكاميرا الدرون في Gazebo
# CAMERA_TOPIC = '/world/default/model/x500_gimbal_0/link/camera_link/sensor/camera/image'
# SPIRAL_RADII = [5, 10, 15, 20]
# # ----------------------------------------

# object_detected = False
# cv_enabled = False  
# model = YOLO('yolov8n.pt') # تحميل موديل YOLO

# # -------- ROS 2 AI NODE --------
# class DroneVisionNode(Node):
#     def __init__(self, drone):
#         super().__init__('drone_vision_node')
#         self.drone = drone
#         self.bridge = CvBridge()
#         self.subscription = self.create_subscription(
#             Image,
#             CAMERA_TOPIC,
#             self.image_callback,
#             10)
#         print("🚀 YOLO AI Node is ready and listening...")

#     def image_callback(self, msg):
#         global object_detected, cv_enabled
        
#         # تحويل الصورة للبث الحي دائماً
#         frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#         annotated_frame = frame.copy()

#         # الكشف عن الشخص فقط إذا بدأ البحث ولم نكتشف أحداً بعد
#         if cv_enabled and not object_detected:
#             # استخدام إعدادات مخففة imgsz=320 لمنع مشكلة "Killed"
#             results = model(frame, stream=True, conf=0.4, verbose=False, imgsz=320)
            
#             for r in results:
#                 annotated_frame = r.plot() 
#                 for box in r.boxes:
#                     if int(box.cls[0]) == 0: # 0 هو كود الإنسان
#                         print("🎯 TARGET DETECTED BY YOLO!")
#                         object_detected = True
#                         cv2.imwrite("detected_person.jpg", annotated_frame)
#                         asyncio.run_coroutine_threadsafe(save_detection_gps(self.drone), loop)

#         # عرض الفيديو للجنة (لن يتوقف البث)
#         cv2.imshow("Rescue Live Stream", annotated_frame)
#         cv2.waitKey(1)

# # -------- DRONE HELPERS (The functions you wanted to keep) --------

# async def wait_one_orbit(drone: System, center_lat: float, center_lon: float, radius_m: float,
#                          circle_tolerance_m: float = 1.5,
#                          start_tolerance_deg: float = 12 * 1e-6):
#     """
#     تنتظر الدرون لإكمال دورة كاملة (360 درجة) حول الهدف
#     """
#     def distance_m(lat1, lon1, lat2, lon2):
#         R = 6371000
#         phi1, phi2 = math.radians(lat1), math.radians(lat2)
#         dphi = math.radians(lat2 - lat1)
#         dlambda = math.radians(lon2 - lon1)
#         a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
#         return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

#     print(f"Waiting for drone to complete orbit at {radius_m}m...")
    
#     # 1. انتظار الوصول لمحيط الدائرة
#     start_lat, start_lon = 0, 0
#     async for pos in drone.telemetry.position():
#         if abs(distance_m(center_lat, center_lon, pos.latitude_deg, pos.longitude_deg) - radius_m) <= circle_tolerance_m:
#             start_lat, start_lon = pos.latitude_deg, pos.longitude_deg
#             print(f"Orbit started at point: {start_lat}, {start_lon}")
#             break

#     # 2. انتظار الابتعاد عن نقطة البداية
#     async for pos in drone.telemetry.position():
#         if object_detected: return # الخروج إذا وجدنا الشخص أثناء الدوران
#         if abs(pos.latitude_deg - start_lat) > start_tolerance_deg or abs(pos.longitude_deg - start_lon) > start_tolerance_deg:
#             break

#     # 3. انتظار العودة لنفس النقطة (دورة كاملة)
#     async for pos in drone.telemetry.position():
#         if object_detected: return
#         if abs(pos.latitude_deg - start_lat) <= start_tolerance_deg and abs(pos.longitude_deg - start_lon) <= start_tolerance_deg:
#             print(f"✅ Full orbit at {radius_m}m completed!")
#             break

# async def save_detection_gps(drone):
#     async for pos in drone.telemetry.position():
#         print(f"📍 VICTIM CAPTURED → LAT:{pos.latitude_deg}, LON:{pos.longitude_deg}")
#         break

# # -------- SPIRAL SEARCH --------
# async def spiral_search(drone, lat, lon, alt):
#     global cv_enabled, object_detected
#     print("🔍 Autonomous Spiral Search Started")
#     cv_enabled = True # تفعيل الكاميرا والـ YOLO

#     for radius in SPIRAL_RADII:
#         if object_detected: break
        
#         print(f"🔄 Orbiting Radius: {radius}m")
#         await drone.action.do_orbit(
#             radius_m=radius,
#             velocity_ms=1.5,
#             yaw_behavior=OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER,
#             latitude_deg=lat,
#             longitude_deg=lon,
#             absolute_altitude_m=alt
#         )

#         # استدعاء الدالة التي طلبتِ الحفاظ عليها
#         await wait_one_orbit(drone, lat, lon, radius)

#     print("✅ Search Phase Finished")

# # -------- MAIN EXECUTION --------
# async def main():
#     global loop
#     loop = asyncio.get_running_loop()
    
#     rclpy.init()
#     drone = System()
#     await drone.connect(system_address="udp://:14540")
    
#     # تشغيل رؤية الكمبيوتر (ROS 2) في خيط منفصل
#     vision_node = DroneVisionNode(drone)
#     asyncio.create_task(asyncio.to_thread(rclpy.spin, vision_node))

#     print("🔓 Arming and Taking off...")
#     await drone.action.arm()
#     await drone.action.takeoff()
#     await asyncio.sleep(8)

#     home = await anext(drone.telemetry.home())
#     target_lat = home.latitude_deg + (11.11 * 0.000009)
#     target_lon = home.longitude_deg + (15.00 * 0.000009)

#     print("✈️ Proceeding to search area...")
#     await drone.action.goto_location(target_lat, target_lon, home.absolute_altitude_m + 5, 0)
#     await asyncio.sleep(10)

#     # بدء البحث الحلزوني المطور
#     await spiral_search(drone, target_lat, target_lon, home.absolute_altitude_m + 5)

#     if object_detected:
#         print("🏠 Success! Returning to base.")
#         await drone.action.return_to_launch()
    
#     await asyncio.sleep(15)
#     rclpy.shutdown()

# if __name__ == "__main__":
#     asyncio.run(main())






#///////////////////////////////////////////////////////////////////////////



import asyncio
import math
import cv2
import numpy as np
import rclpy
import requests  
import threading
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
from mavsdk.action import OrbitYawBehavior

# ---------------- CONFIG ----------------
CAMERA_TOPIC = '/world/default/model/x500_gimbal_0/link/camera_link/sensor/camera/image'
SPIRAL_RADII = [5, 10, 15, 20]

API_URL = "https://v0-sky-guard-drone-dashboard.vercel.app/api/detection"
# ----------------------------------------

object_detected = False
cv_enabled = False  
model = YOLO('yolov8n.pt') 

def upload_to_web(lat, lon, image_path):
    try:
        payload = {"lat": str(lat), "lon": str(lon), "status": "Person Detected"}
        with open(image_path, "rb") as img:
            files = {"image": img}
            response = requests.post(API_URL, data=payload, files=files, timeout=10)
            print(f"📡 Web Update: Success ({response.status_code})")
    except Exception as e:
        print(f"📡 Web Update: Failed ({e})")

# -------- ROS 2 AI NODE --------
class DroneVisionNode(Node):
    def __init__(self, drone):
        super().__init__('drone_vision_node')
        self.drone = drone
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10)
        print("🚀 YOLO AI Node is ready and listening...")

    def image_callback(self, msg):
        global object_detected, cv_enabled
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        annotated_frame = frame.copy()

        if cv_enabled and not object_detected:
            results = model(frame, stream=True, conf=0.4, verbose=False, imgsz=320)
            
            for r in results:
                annotated_frame = r.plot() 
                for box in r.boxes:
                    if int(box.cls[0]) == 0: 
                        print("🎯 TARGET DETECTED!")
                        object_detected = True
                        image_path = "detected_person.jpg"
                        cv2.imwrite(image_path, annotated_frame)
                        
                        asyncio.run_coroutine_threadsafe(self.process_detection(image_path), loop)

        cv2.imshow("Rescue Live Stream", annotated_frame)
        cv2.waitKey(1)

    async def process_detection(self, image_path):
        # 1. الحصول على الإحداثيات من الدرون
        async for pos in self.drone.telemetry.position():
            lat, lon = pos.latitude_deg, pos.longitude_deg
            print(f"📍 Location Captured: {lat}, {lon}")
            
            # 2. إرسال البيانات للموقع في خيط منفصل (Thread) لكي لا يتوقف الدرون
            threading.Thread(target=upload_to_web, args=(lat, lon, image_path)).start()
            break

# -------- بقية الدوال (بدون تغيير لضمان الاستقرار) --------

async def wait_one_orbit(drone: System, center_lat: float, center_lon: float, radius_m: float,
                         circle_tolerance_m: float = 1.5,
                         start_tolerance_deg: float = 12 * 1e-6):
    def distance_m(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    print(f"Waiting for drone to complete orbit at {radius_m}m...")
    start_lat, start_lon = 0, 0
    async for pos in drone.telemetry.position():
        if abs(distance_m(center_lat, center_lon, pos.latitude_deg, pos.longitude_deg) - radius_m) <= circle_tolerance_m:
            start_lat, start_lon = pos.latitude_deg, pos.longitude_deg
            break

    async for pos in drone.telemetry.position():
        if object_detected: return 
        if abs(pos.latitude_deg - start_lat) > start_tolerance_deg or abs(pos.longitude_deg - start_lon) > start_tolerance_deg:
            break

    async for pos in drone.telemetry.position():
        if object_detected: return
        if abs(pos.latitude_deg - start_lat) <= start_tolerance_deg and abs(pos.longitude_deg - start_lon) <= start_tolerance_deg:
            print(f"✅ Orbit {radius_m}m completed!")
            break

async def spiral_search(drone, lat, lon, alt):
    global cv_enabled, object_detected
    print("🔍 Autonomous Spiral Search Started")
    cv_enabled = True 

    for radius in SPIRAL_RADII:
        if object_detected: break
        print(f"🔄 Orbiting Radius: {radius}m")
        await drone.action.do_orbit(radius_m=radius, velocity_ms=1.5, 
                                   yaw_behavior=OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER,
                                   latitude_deg=lat, longitude_deg=lon, absolute_altitude_m=alt)
        await wait_one_orbit(drone, lat, lon, radius)

async def main():
    global loop
    loop = asyncio.get_running_loop()
    rclpy.init()
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    vision_node = DroneVisionNode(drone)
    asyncio.create_task(asyncio.to_thread(rclpy.spin, vision_node))

    print("🔓 Arming and Taking off...")
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(8)

    home = await anext(drone.telemetry.home())
    target_lat = home.latitude_deg + (11.11 * 0.000009)
    target_lon = home.longitude_deg + (15.00 * 0.000009)

    await drone.action.goto_location(target_lat, target_lon, home.absolute_altitude_m + 5, 0)
    await asyncio.sleep(10)
    await spiral_search(drone, target_lat, target_lon, home.absolute_altitude_m + 5)

    if object_detected:
        print("🏠 Returning to base...")
        await drone.action.return_to_launch()
    
    await asyncio.sleep(15)
    rclpy.shutdown()

if __name__ == "__main__":
    asyncio.run(main())