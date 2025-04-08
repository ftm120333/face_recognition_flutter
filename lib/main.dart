import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'ML/Recognition.dart';
import 'ML/Recognizer.dart';

late List<CameraDescription> cameras;
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key? key}) : super(key: key);
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  dynamic controller;
  bool isBusy = false;
  late Size size;
  late CameraDescription description = cameras[1];
  CameraLensDirection camDirec = CameraLensDirection.front;
  late List<Recognition> recognitions = [];
  int samplesCollected = 0;
  List<List<double>> sampleEmbeddings = [];

  //TODO declare face detector
  late FaceDetector faceDetector;

  //TODO declare face recognizer
  late Recognizer recognizer;

  @override
  void initState() {
    super.initState();
    var options =
        FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate);
    faceDetector = FaceDetector(options: options);
    recognizer = Recognizer();
    initializeCamera();
  }

  initializeCamera() async {
    controller = CameraController(description, ResolutionPreset.medium,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420 // for Android
            : ImageFormatGroup.bgra8888,
        enableAudio: false); // for iOS);
    await controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {
        controller;
      });
      controller.startImageStream((image) => {
            if (!isBusy)
              {isBusy = true, frame = image, doFaceDetectionOnFrame()}
          });
    });
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  //face detection on a frame

  dynamic _scanResults;
  CameraImage? frame;
  doFaceDetectionOnFrame() async {
    InputImage? inputImage = getInputImage();
    List<Face> faces = await faceDetector.processImage(inputImage!);
    for (Face face in faces) {
      print("Face detected: ${face.boundingBox}");
    }
    performFaceRecognition(faces);
  }

  img.Image? image;
  bool register = false;
  performFaceRecognition(List<Face> faces) async {
    recognitions.clear();

    if (Platform.isIOS) {
      image = _convertBGRA8888ToImage(frame!);
    } else {
      image =
          convertYUV420ToImage(frame!); // Use your YUV420_888 Conversion method
    }
    // creates a new image that contains only the face region.
    image = img.copyRotate(image!,
        angle: camDirec == CameraLensDirection.front ? 270 : 90);

    // image = Platform.isIOS
    //     ? _convertBGRA8888ToImage(frame!) as img.Image?
    //     : _convertNV21(frame!);
    // image = img.copyRotate(image!,
    //     angle: camDirec == CameraLensDirection.front ? 270 : 90);

    for (Face face in faces) {
      Rect faceRect = face.boundingBox;
      //crop face
      img.Image croppedFace = img.copyCrop(image!,
          x: faceRect.left.toInt(),
          y: faceRect.top.toInt(),
          width: faceRect.width.toInt(),
          height: faceRect.height.toInt());

      //pass cropped face to face recognition model
      Recognition recognition = recognizer.recognize(croppedFace, faceRect);
      // if (recognition.distance > 0.6) {
      //   recognition.name = "Unknown";
      // }
      if (recognition.distance > 1.0) {
        recognition.name = "Unknown";
      }
      recognitions.add(recognition);

      //show face registration dialogue
      if (register) {
        showFaceRegistrationDialogue(croppedFace, recognition);
        register = false;
      }
    }

    setState(() {
      isBusy = false;
      _scanResults = recognitions;
    });
  }

  // Face Registration Dialogue
  TextEditingController textEditingController = TextEditingController();
  // showFaceRegistrationDialogue(img.Image croppedFace, Recognition recognition) {
  //   showDialog(
  //     context: context,
  //     builder: (ctx) => AlertDialog(
  //       title: const Text("Face Registration", textAlign: TextAlign.center),
  //       alignment: Alignment.center,
  //       content: SizedBox(
  //         height: 340,
  //         child: Column(
  //           crossAxisAlignment: CrossAxisAlignment.center,
  //           children: [
  //             const SizedBox(
  //               height: 20,
  //             ),
  //             Image.memory(
  //               Uint8List.fromList(img.encodePng(croppedFace)),
  //               width: 200,
  //               height: 200,
  //             ),
  //             SizedBox(
  //               width: 200,
  //               child: TextField(
  //                   controller: textEditingController,
  //                   decoration: const InputDecoration(
  //                       fillColor: Colors.white,
  //                       filled: true,
  //                       hintText: "Enter Name")),
  //             ),
  //             const SizedBox(
  //               height: 10,
  //             ),
  //             ElevatedButton(
  //                 onPressed: () {
  //                   recognizer.registerFaceInDB(
  //                       textEditingController.text, recognition.embeddings);
  //                   textEditingController.text = "";
  //                   Navigator.pop(context);
  //                   ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
  //                     content: Text("Face Registered"),
  //                   ));
  //                 },
  //                 style: ElevatedButton.styleFrom(
  //                     backgroundColor: Colors.blue,
  //                     minimumSize: const Size(200, 40)),
  //                 child: const Text("Register"))
  //           ],
  //         ),
  //       ),
  //       contentPadding: EdgeInsets.zero,
  //     ),
  //   );
  // }

// Replace showFaceRegistrationDialogue with this:

  void showFaceRegistrationDialogue(
      img.Image croppedFace, Recognition recognition) {
    samplesCollected++;
    sampleEmbeddings.add(recognition.embeddings);

    if (samplesCollected < 3) {
      // Collect 3 samples
      showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          title:
              Text("Sample $samplesCollected/3", textAlign: TextAlign.center),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Image.memory(
                Uint8List.fromList(img.encodePng(croppedFace)),
                width: 200,
                height: 200,
              ),
              SizedBox(height: 20),
              Text("Please move your head slightly",
                  style: TextStyle(fontSize: 16)),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: Text("Capture Next Sample"),
              ),
            ],
          ),
        ),
      );
    } else {
      // After collecting all samples, show name input
      showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text("Final Registration", textAlign: TextAlign.center),
          content: SizedBox(
            height: 340,
            child: Column(
              children: [
                Text("3 samples collected", style: TextStyle(fontSize: 16)),
                SizedBox(height: 20),
                TextField(
                  controller: textEditingController,
                  decoration: InputDecoration(hintText: "Enter Name"),
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () async {
                    // Average all samples
                    List<double> averaged =
                        List.filled(sampleEmbeddings[0].length, 0.0);
                    for (var embedding in sampleEmbeddings) {
                      for (int i = 0; i < embedding.length; i++) {
                        averaged[i] += embedding[i];
                      }
                    }
                    for (int i = 0; i < averaged.length; i++) {
                      averaged[i] /= sampleEmbeddings.length;
                    }

                    await recognizer.registerFaceInDB(
                        textEditingController.text, averaged);

                    textEditingController.text = "";
                    samplesCollected = 0;
                    sampleEmbeddings.clear();
                    Navigator.pop(context);
                  },
                  child: Text("Register"),
                ),
              ],
            ),
          ),
        ),
      );
    }
  }

  static var IOS_BYTES_OFFSET = 28;

  static img.Image _convertBGRA8888ToImage(CameraImage cameraImage) {
    final plane = cameraImage.planes[0];

    return img.Image.fromBytes(
      width: cameraImage.width,
      height: cameraImage.height,
      bytes: plane.bytes.buffer,
      rowStride: plane.bytesPerRow,
      bytesOffset: IOS_BYTES_OFFSET,
      order: img.ChannelOrder.bgra,
    );
  }

  // Converts NV21 CameraImage to a usable RGB image (img.Image)
  static img.Image _convertNV21(CameraImage image) {
    final width = image.width.toInt();
    final height = image.height.toInt();

    Uint8List yuv420sp = image.planes[0].bytes;

    final outImg = img.Image(height: height, width: width);
    final int frameSize = width * height;

    for (int j = 0, yp = 0; j < height; j++) {
      int uvp = frameSize + (j >> 1) * width, u = 0, v = 0;
      for (int i = 0; i < width; i++, yp++) {
        int y = (0xff & yuv420sp[yp]) - 16;
        if (y < 0) y = 0;
        if ((i & 1) == 0) {
          v = (0xff & yuv420sp[uvp++]) - 128;
          u = (0xff & yuv420sp[uvp++]) - 128;
        }
        int y1192 = 1192 * y;
        int r = (y1192 + 1634 * v);
        int g = (y1192 - 833 * v - 400 * u);
        int b = (y1192 + 2066 * u);

        if (r < 0)
          r = 0;
        else if (r > 262143) r = 262143;
        if (g < 0)
          g = 0;
        else if (g > 262143) g = 262143;
        if (b < 0)
          b = 0;
        else if (b > 262143) b = 262143;

        // I don't know how these r, g, b values are defined, I'm just copying what you had bellow and
        // getting their 8-bit values.
        outImg.setPixelRgb(i, j, ((r << 6) & 0xff0000) >> 16,
            ((g >> 2) & 0xff00) >> 8, (b >> 10) & 0xff);
      }
    }
    return outImg;
  }

  // method to convert CameraImage to Image
  img.Image convertYUV420ToImage(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;

    final yRowStride = cameraImage.planes[0].bytesPerRow;
    final uvRowStride = cameraImage.planes[1].bytesPerRow;
    final uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

    final image = img.Image(width: width, height: height);

    for (var w = 0; w < width; w++) {
      for (var h = 0; h < height; h++) {
        final uvIndex =
            uvPixelStride * (w / 2).floor() + uvRowStride * (h / 2).floor();
        final index = h * width + w;
        final yIndex = h * yRowStride + w;

        final y = cameraImage.planes[0].bytes[yIndex];
        final u = cameraImage.planes[1].bytes[uvIndex];
        final v = cameraImage.planes[2].bytes[uvIndex];

        image.data!.setPixelR(w, h, yuv2rgb(y, u, v)); //= yuv2rgb(y, u, v);
      }
    }
    return image;
  }

  //Converts YUV420 (camera format) to raw NV21 byte array
  Uint8List? convertYUV420ToNV21(CameraImage image) {
    try {
      final int width = image.width;
      final int height = image.height;
      final int ySize = width * height;
      final int uvSize = ySize ~/ 4;

      final Uint8List nv21 = Uint8List(ySize + 2 * uvSize);
      final Uint8List y = image.planes[0].bytes;
      final Uint8List u = image.planes[1].bytes;
      final Uint8List v = image.planes[2].bytes;

      // Copy Y channel (Luma)
      for (int i = 0; i < height; i++) {
        nv21.setRange(
            i * width,
            (i * width) + width,
            y.sublist(i * image.planes[0].bytesPerRow,
                (i * image.planes[0].bytesPerRow) + width));
      }

      // Interleave U and V channels (Chroma)
      int uvIndex = ySize;
      for (int i = 0; i < height ~/ 2; i++) {
        for (int j = 0; j < width ~/ 2; j++) {
          nv21[uvIndex++] = v[i * image.planes[2].bytesPerRow +
              j * image.planes[2].bytesPerPixel!]; // V
          nv21[uvIndex++] = u[i * image.planes[1].bytesPerRow +
              j * image.planes[1].bytesPerPixel!]; // U
        }
      }

      return nv21;
    } catch (e) {
      print("❌ Error converting YUV to NV21: $e");
      return null;
    }
  }

  int yuv2rgb(int y, int u, int v) {
    // Convert yuv pixel to rgb
    var r = (y + v * 1436 / 1024 - 179).round();
    var g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
    var b = (y + u * 1814 / 1024 - 227).round();

    // Clipping RGB values to be inside boundaries [ 0 , 255 ]
    r = r.clamp(0, 255);
    g = g.clamp(0, 255);
    b = b.clamp(0, 255);

    return 0xff000000 |
        ((b << 16) & 0xff0000) |
        ((g << 8) & 0xff00) |
        (r & 0xff);
  }

  final _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  InputImage? getInputImage() {
    if (controller == null || !controller!.value.isInitialized) {
      return null;
    }

    if (frame == null) {
      return null;
    }
    // Always using the front camera
    final camera = cameras[1]; // Front camera
    final sensorOrientation = camera.sensorOrientation;

    // Calculate rotation compensation
    var rotationCompensation =
        _orientations[controller!.value.deviceOrientation];
    if (rotationCompensation == null) {
      return null;
    }

    // Adjust rotation for front camera
    rotationCompensation = (sensorOrientation + rotationCompensation) % 360;

    // Get the rotation value
    InputImageRotation? rotation =
        InputImageRotationValue.fromRawValue(rotationCompensation);
    if (rotation == null) {
      return null;
    }

    // Convert YUV_420_888 (format 35) to NV21
    print("🔹 Detected format: ${frame!.format.raw}");
    Uint8List? convertedBytes;
    if (frame!.format.raw == 35) {
      // YUV_420_888
      convertedBytes = convertYUV420ToNV21(frame!);
      if (convertedBytes == null) {
        return null;
      }
    } else {
      return null;
    }

    return InputImage.fromBytes(
      bytes: convertedBytes, // Use the converted NV21 bytes
      metadata: InputImageMetadata(
        size: Size(frame!.width.toDouble(), frame!.height.toDouble()),
        rotation: rotation,
        format: InputImageFormat.nv21, // Force NV21 format
        bytesPerRow: frame!.width, // Ensure correct row alignment
      ),
    );
  }

  Widget buildResult() {
    if (_scanResults == null ||
        controller == null ||
        !controller.value.isInitialized) {
      return const Center(child: Text('Camera is not initialized  '));
    }
    final Size imageSize = Size(
      controller.value.previewSize!.height,
      controller.value.previewSize!.width,
    );
    CustomPainter painter =
        FaceDetectorPainter(imageSize, _scanResults, camDirec);
    return CustomPaint(
      painter: painter,
    );
  }

  // toggle camera direction
  void _toggleCameraDirection() async {
    if (camDirec == CameraLensDirection.back) {
      camDirec = CameraLensDirection.front;
      description = cameras[1];
    } else {
      camDirec = CameraLensDirection.back;
      description = cameras[0];
    }
    await controller.stopImageStream();
    setState(() {
      controller;
    });

    initializeCamera();
  }

  @override
  Widget build(BuildContext context) {
    List<Widget> stackChildren = [];
    size = MediaQuery.of(context).size;
    if (controller != null) {
      //TODO View for displaying the live camera footage
      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          width: size.width,
          height: size.height,
          child: Container(
            child: (controller.value.isInitialized)
                ? AspectRatio(
                    aspectRatio: controller.value.aspectRatio,
                    child: CameraPreview(controller),
                  )
                : Container(),
          ),
        ),
      );

      //TODO View for displaying rectangles around detected aces
      stackChildren.add(
        Positioned(
            top: 0.0,
            left: 0.0,
            width: size.width,
            height: size.height,
            child: buildResult()),
      );
    }

    //TODO View for displaying the bar to switch camera direction or for registering faces
    stackChildren.add(Positioned(
      top: size.height - 140,
      left: 0,
      width: size.width,
      height: 80,
      child: Card(
        margin: const EdgeInsets.only(left: 20, right: 20),
        color: Colors.blue,
        child: Center(
          child: Container(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    IconButton(
                      icon: const Icon(
                        Icons.cached,
                        color: Colors.white,
                      ),
                      iconSize: 40,
                      color: Colors.black,
                      onPressed: () {
                        _toggleCameraDirection();
                      },
                    ),
                    Container(
                      width: 30,
                    ),
                    IconButton(
                      icon: const Icon(
                        Icons.face_retouching_natural,
                        color: Colors.white,
                      ),
                      iconSize: 40,
                      color: Colors.black,
                      onPressed: () {
                        setState(() {
                          register = true;
                        });
                      },
                    )
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    ));

    return SafeArea(
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Container(
            margin: const EdgeInsets.only(top: 0),
            color: Colors.black,
            child: Stack(
              children: stackChildren,
            )),
      ),
    );
  }
}

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(this.absoluteImageSize, this.faces, this.camDire2);

  final Size absoluteImageSize;
  final List<Recognition> faces;
  CameraLensDirection camDire2;

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / absoluteImageSize.width;
    final double scaleY = size.height / absoluteImageSize.height;

    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.indigoAccent;

    for (Recognition face in faces) {
      canvas.drawRect(
        Rect.fromLTRB(
          camDire2 == CameraLensDirection.front
              ? (absoluteImageSize.width - face.location.right) * scaleX
              : face.location.left * scaleX,
          face.location.top * scaleY,
          camDire2 == CameraLensDirection.front
              ? (absoluteImageSize.width - face.location.left) * scaleX
              : face.location.right * scaleX,
          face.location.bottom * scaleY,
        ),
        paint,
      );

      TextSpan span = TextSpan(
          style: const TextStyle(color: Colors.white, fontSize: 20),
          text: "${face.name}  ${face.distance.toStringAsFixed(2)}");
      TextPainter tp = TextPainter(
          text: span,
          textAlign: TextAlign.left,
          textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(canvas,
          Offset(face.location.left * scaleX, face.location.top * scaleY));
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return true;
  }
}
