import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  OrtEnv.instance.init();

  final cameras = await availableCameras();
  final backCamera = cameras[0];

  runApp(MyApp(camera: backCamera));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key, required this.camera});

  final CameraDescription camera;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: MyHomePage(camera: camera),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.camera});

  final CameraDescription camera;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late CameraController _cameraController;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();

    _cameraController = CameraController(widget.camera, ResolutionPreset.medium, enableAudio: false);

    _initializeControllerFuture = _cameraController.initialize();
  }

  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Take a picture')),
      body: FutureBuilder(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Center(child: CameraPreview(_cameraController));
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          try {
            await _initializeControllerFuture;
            final XFile image = await _cameraController.takePicture();

            final directory = await getApplicationDocumentsDirectory();
            final imagePath = join(directory.path, '${DateTime.now()}.png');
            await image.saveTo(imagePath);

            if (!context.mounted) {
              return;
            }
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => DisplayPictureScreen(imagePath: imagePath),
              ),
            );
          } catch (e) {
            print(e);
          }
        },
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}

class DisplayPictureScreen extends StatefulWidget {
  final String imagePath;

  const DisplayPictureScreen({super.key, required this.imagePath});

  @override
  State<DisplayPictureScreen> createState() => _DisplayPictureScreenState();
}

class _DisplayPictureScreenState extends State<DisplayPictureScreen> {
  OutputBoxes? boxes;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Display the Picture')),
      body: Column(
        children: [
          if (boxes != null)
            Container(
              height: 320,
              width: 240,
              color: Colors.red,
              child: LayoutBuilder(builder: (context, constraints) {
                final scaleX = constraints.maxWidth / boxes!.originalWidth;
                final scaleY = constraints.maxHeight / boxes!.originalHeight;

                return Stack(
                  children: [
                    Image.file(File(widget.imagePath)),
                    ...boxes!.outputBoxes.map((box) {
                      return Positioned(
                        left: (box.centerX - (box.width / 2)) * scaleX,
                        top: (box.centerY - (box.height / 2)) * scaleY,
                        width: box.width * scaleX,
                        height: box.height * scaleY,
                        child: Container(
                          decoration: BoxDecoration(
                            border: Border.all(color: Colors.red),
                          ),
                        ),
                      );
                    })
                  ],
                );
              }),
            ),
          if (boxes == null)
            SizedBox(
              height: 320,
              width: 240,
              child: Image.file(File(widget.imagePath)),
            ),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () async {
                final image = await _readImage(widget.imagePath);
                await _analyze(image);
              },
              child: const Text('Run Detection'),
            ),
          ),
        ],
      ),
    );
  }

  Future<img.Image> _readImage(String imagePath) async {
    final file = File(imagePath);
    final imageBytes = await file.readAsBytes();
    final image = img.decodeImage(imageBytes)!;
    return image;
  }

  Future<void> _analyze(img.Image image) async {
    // Creating the session
    final session = await _createSession();

    // Preprocessing the image
    final preprocessedInput = _preprocessImage(image);

    // Preparing inputs
    final inputs = _createInputs(image: image, preprocessedInput: preprocessedInput);

    // Running inference
    final runOptions = OrtRunOptions();
    final outputs = await session.runAsync(runOptions, inputs);

    // Processing outputs
    final outputBoxes = OutputBoxes.from(outputs, originalWidth: image.width, originalHeight: image.height);
    if (outputBoxes != null) {
      setState(() {
        boxes = outputBoxes;
      });
    }

    // Clean Up
    for (final input in inputs.values) {
      input.release();
    }
    runOptions.release();
  }

  Future<OrtSession> _createSession() async {
    final sessionOptions = OrtSessionOptions();
    const modelFileName = 'yolov5n.onnx';
    final modelFile = await rootBundle.load(modelFileName);
    final bytes = modelFile.buffer.asUint8List();
    final session = OrtSession.fromBuffer(bytes, sessionOptions);
    return session;
  }

  Float32List _preprocessImage(img.Image image) {
    // Resizing the image
    const inputWidth = 640, inputHeight = 640;
    final outputImage = img.copyResize(image, width: inputWidth, height: inputHeight);

    // Transposing from [r1,g1,b1,r2,g2,b2,...] to [r1,r2,...,g1,g2,...,b1,b2]
    List<int> redValues = [];
    List<int> greenValues = [];
    List<int> blueValue = [];
    for (final pixel in outputImage) {
      redValues.add(pixel.r.toInt());
      greenValues.add(pixel.g.toInt());
      blueValue.add(pixel.b.toInt());
    }
    final transposedValues = [...redValues, ...greenValues, ...blueValue];

    // Normalize the pixel values to [0, 1] range
    List<double> normalizedValues = transposedValues.map((value) => value / 255.0).toList();

    return Float32List.fromList(normalizedValues);
  }

  Map<String, OrtValueTensor> _createInputs({required img.Image image, required Float32List preprocessedInput}) {
    final imagesTensorShape = [1, 3, 640, 640];
    final imagesTensor = OrtValueTensor.createTensorWithDataList(preprocessedInput, imagesTensorShape);

    final inputs = {'images': imagesTensor};
    return inputs;
  }
}

class OutputBoxes {
  OutputBoxes({
    required this.outputBoxes,
    required this.originalWidth,
    required this.originalHeight,
  });

  static OutputBoxes? from(List<OrtValue?>? outputs, {required int originalWidth, required int originalHeight}) {
    // Checking if the outputs are there and if there are exactly 1 of them
    if (outputs == null || outputs.length != 1) {
      print('Incorrect outputs');
      return null;
    }

    // All boxes found by the model as 3D double array [1 x num_boxes x 4]
    final results = outputs[0]?.value;
    if (results == null || results is! List<List<List<double>>>) {
      print('Incorrect boxes');
      return null;
    }

    // Iterating over the results
    List<OutputBox> outputBoxes = [];
    for (final result in results[0]) {
      final score = result[5];
      const threshold = 0.9;
      if (score < threshold) {
        continue;
      }
      final centerX = ((result[0] / 640.0) * originalWidth).toInt();
      final centerY = ((result[1] / 640.0) * originalHeight).toInt();
      final width = ((result[2] / 640.0) * originalWidth).toInt();
      final height = ((result[3] / 640.0) * originalHeight).toInt();
      final outputBox = OutputBox(width: width, height: height, centerX: centerX, centerY: centerY, score: score);
      outputBoxes.add(outputBox);
    }

    // Non maximum suppression
    outputBoxes = _nonMaxSuppression(outputBoxes);

    return OutputBoxes(outputBoxes: outputBoxes, originalWidth: originalWidth, originalHeight: originalHeight);
  }

  static List<OutputBox> _nonMaxSuppression(List<OutputBox> boxes) {
    // Maximum overlap between two boxes threshold
    const threshold = 0.15;

    final indices = [for (var index = 0; index < boxes.length; ++index) index];

    // Sorting the boxes based on the confidence score from high to low
    final sortedIndices = [...indices]..sort((a, b) => boxes[b].score.compareTo(boxes[a].score));

    // Iterating over all the boxes and deciding which ones to keep
    List<int> selectedBoxesIndices = [];
    for (var index = 0; index < sortedIndices.length; ++index) {
      var shouldSelect = true;
      final box = boxes[sortedIndices[index]];

      // Iterating over already selected boxes to check if the new box does not overlap too much
      for (var selectedBoxIndex = 0; selectedBoxIndex < selectedBoxesIndices.length; ++selectedBoxIndex) {
        final selectedBox = boxes[selectedBoxesIndices[selectedBoxIndex]];
        final intersectionArea = _intersectionOverUnion(box, selectedBox);
        if (intersectionArea > threshold) {
          shouldSelect = false;
          break;
        }
      }

      // The current box does not overlap too much and is selected
      if (shouldSelect) {
        selectedBoxesIndices.add(sortedIndices[index]);
      }
    }

    return selectedBoxesIndices.map((index) => boxes[index]).toList();
  }

  static double _intersectionOverUnion(OutputBox a, OutputBox b) {
    final areaA = a.width * a.height, areaB = b.width * b.height;
    if (areaA <= 0.0 || areaB <= 0.0) {
      return 0.0;
    }

    final aMinX = a.centerX - (a.width / 2), bMinX = b.centerX - (b.width / 2);
    final aMaxX = a.centerX + (a.width / 2), bMaxX = b.centerX + (b.width / 2);
    final aMinY = a.centerY - (a.height / 2), bMinY = b.centerY - (b.height / 2);
    final aMaxY = a.centerY + (a.height / 2), bMaxY = b.centerY + (b.height / 2);

    final intersectionMinX = max(aMinX, bMinX);
    final intersectionMaxX = min(aMaxX, bMaxX);
    final intersectionMinY = max(aMinY, bMinY);
    final intersectionMaxY = min(aMaxY, bMaxY);

    final intersectionArea = max(intersectionMaxY - intersectionMinY, 0) * max(intersectionMaxX - intersectionMinX, 0);

    return intersectionArea / (areaA + areaB - intersectionArea);
  }

  final List<OutputBox> outputBoxes;
  final int originalWidth;
  final int originalHeight;
}

class OutputBox {
  OutputBox({
    required this.width,
    required this.height,
    required this.centerX,
    required this.centerY,
    required this.score,
  });

  final int width;
  final int height;
  final int centerX;
  final int centerY;
  final double score;
}
