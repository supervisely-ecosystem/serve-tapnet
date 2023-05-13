<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/serve-tapnet/assets/115161827/967a413a-afb9-4051-afe7-ff740bea1bf5" />
  
# TAP-Net object tracking

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-tapnet/tapnet/supervisely/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-tapnet)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/pips/supervisely/serve-tapnet/tapnet/supervisely/serve)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/pips/supervisely/serve-tapnet/tapnet/supervisely/serve)](https://supervise.ly)


</div>

# Overview

This app is an integration of TAP-Net model, which is a NN-assisted interactive object tracking model. The TAP-Net model can track point trajectories. It is used to implement the tracking of polygons, points and graphs (keypoints) on videos.

# How to Run

0. Run the application from Ecosystem

1. Open Video Labeling interface

2. Configure tracking settings

3. Press `Track` button

https://user-images.githubusercontent.com/115161827/237701946-2ef6b5bd-3473-4df6-bf7d-33539377c429.mp4

4. After finishing working with the app, stop the app session manually in the `App sessions` tab

# Keypoints tracking example

You can also use this app to track keypoints. This app can track keypoints graph of any shape and number of points.

1. Open your video project, select suitable frame and click on "Screenshot" button in the upper right corner:

https://user-images.githubusercontent.com/91027877/238152827-1a6fcc7b-7d68-4168-86af-7406d6255d9c.mp4


2. Create keypoints class based on your screenshot:

https://user-images.githubusercontent.com/91027877/238153794-43870be8-37bd-434a-bdf7-536da5267602.mp4

3. Go back to video, set your recently created keypoints graph on target object, select number of frames to be tracked and click on "Track" button:

https://user-images.githubusercontent.com/91027877/238153954-6364579b-2dff-49c4-b4da-35d4ea0e9ce9.mp4


You can change visualization settings of your keypoints graph in right sidebar:

https://user-images.githubusercontent.com/91027877/238154341-ed9acea5-2693-421d-a673-a6f4ab8f515a.mp4


# Acknowledgment

This app is based on the great work `TAP-Net` 
- [GitHub](https://github.com/deepmind/tapnet) ![GitHub Org's stars](https://img.shields.io/github/stars/deepmind/tapnet?style=social) 
- [Paper](https://arxiv.org/abs/2211.03726) 





