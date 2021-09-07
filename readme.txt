README.TXT

본 깃에는 AR 면접 과제 프로젝트 파일 및 시연 영상이 포함되어 있습니다.

camMatrix.npz - 3차원 축 생성을 위한 calibration 데이터입니다. camMat['mat']: camera matrix, camMat['dist']: distortion matrix, camMat['rvecs']: rotation matrix, camMat['tvecs']: translation matrix를 포함합니다..

ArucoMarkerTest.py 	- Aruco마커 인식(초록색 테두리 표시), 사전에 정의된 딕셔너리와 비교하여 ID 확인(시안색 글씨), 각 마커를 중심으로한 3차원 축을 출력합니다.
			- 박동우, 문지원, 정현석, 김영헌, 황성수 (2018). 속도와 시야각에 강인한 다중 ArUco 마커 기반 증강현실 시스템. 한국HCI학회 학술대회, 916-920 <그림2> 알고리즘 참조

ArucoMarketTest2.py	- 테두리 인식 및 축의 출력을 제거하고 정해진 네개의 마커가 같은 방향으로 존재하고 카메라가 이를 인식 할 경우 사전에 정의된 text_drawing을 카메라영상위에 입혀 출력함.

test_drawing.jpg 	- 합성될 영상. 각 모서리에 마커를 입혔으며 동일 마커 네개를 모두 발견할 경우 위에 합성되 출력된다.


output1.mp4		- ArucoMarkerTest 시연영상
output2.mp4		- ArucoMarkerTest2 시연영상
