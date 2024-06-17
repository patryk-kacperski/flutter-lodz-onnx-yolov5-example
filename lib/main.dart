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
                print('Constraints maxWidth: ${constraints.maxWidth}');
                print('Constraints maxHeight: ${constraints.maxHeight}');
                print('Original width: ${boxes!.originalWidth}');
                print('Original height: ${boxes!.originalHeight}');
                print('scaleX: $scaleX');
                print('scaleY: $scaleY');
                print('\n');

                return Stack(
                  children: [
                    Image.file(File(widget.imagePath)),
                    ...boxes!.outputBoxes.map((box) {
                      print('centerX ${box.centerX}');
                      print('centerY ${box.centerY}');
                      print('width ${box.width}');
                      print('height ${box.height}');
                      print('\n');
                      return Positioned(
                        left: (box.centerX - (box.width / 2)) * scaleX,
                        top: (box.centerY - (box.height / 2)) * scaleY,
                        width: box.width * scaleX,
                        height: box.height * scaleY,
                        // left: box.width * scaleX - (box.centerX / 2) * scaleX,
                        // top: box.height * scaleY - (box.centerY / 2) * scaleY,
                        // width: box.centerX * scaleX,
                        // height: box.centerY * scaleY,
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
            Container(
              height: 320,
              width: 240,
              child: Image.file(File(widget.imagePath)),
            ),
          Container(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () async {
                // final pixels = await _convertImageToPixels(imagePath);
                final image = await _readImage(widget.imagePath);
                await _analyze(image);
              },
              child: const Text('Convert to Pixel Array'),
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
    print('Image read: ${image.width} x ${image.height}');
    return image;
  }

  // Future<void> _analyze(Uint8List pixels) async {
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
    // const modelFileName = 'tiny-yolov3-11.ort';
    const modelFileName = 'tiny-yolov3-11.onnx';
    final modelFile = await rootBundle.load(modelFileName);
    final bytes = modelFile.buffer.asUint8List();
    final session = OrtSession.fromBuffer(bytes, sessionOptions);
    return session;
  }

  Float32List _preprocessImage(img.Image image) {
    // Resizing the image
    final imageWidth = image.width, imageHeight = image.height;
    const inputWidth = 416, inputHeight = 416;
    final widthRatio = inputWidth / imageWidth, heightRatio = inputHeight / imageHeight;
    final scale = min(widthRatio, heightRatio);
    final resizedWidth = (imageWidth * scale).toInt(), resizedHeight = (imageHeight * scale).toInt();
    final resizedImage = img.copyResize(image, width: resizedWidth, height: resizedHeight);
    // final resizedImage = img.copyResize(image, width: inputWidth, height: inputHeight);

    // Preparing output image filled with gray
    final outputImage = img.Image(width: inputHeight, height: inputHeight);
    for (var pixel in outputImage) {
      pixel
        ..r = 128
        ..g = 128
        ..b = 128;
    }

    // Resized image offset
    final xOffset = (inputWidth - resizedWidth) ~/ 2;
    final yOffset = (inputHeight - resizedHeight) ~/ 2;

    // Inserting the resized image into the output image
    for (final pixel in resizedImage) {
      final offsetXPosition = xOffset + pixel.x, offsetYPosition = yOffset + pixel.y;
      outputImage.setPixel(offsetXPosition, offsetYPosition, pixel);
    }

    // Transposing from [r1,g1,b1,r2,g2,b2,...] to [r1,r2,...,g1,g2,...,b1,b2]
    List<int> redValues = [];
    List<int> greenValues = [];
    List<int> blueValue = [];
    for (final pixel in outputImage) {
      redValues.add(pixel.r.toInt());
      greenValues.add(pixel.g.toInt());
      blueValue.add(pixel.b.toInt());
    }
    // for (final pixel in resizedImage) {
    //   redValues.add(pixel.r.toInt());
    //   greenValues.add(pixel.g.toInt());
    //   blueValue.add(pixel.b.toInt());
    // }
    final transposedValues = [...redValues, ...greenValues, ...blueValue];
    // final transposedValues = [...blueValue, ...greenValues, ...redValues];
    // final transposedValues = [...greenValues, ...redValues, ...blueValue];

    // Normalize the pixel values to [0, 1] range
    List<double> normalizedValues = transposedValues.map((value) => value / 255.0).toList();

    // List<double> normalizedValues = [];
    // for (final pixel in outputImage) {
    //   normalizedValues
    //     ..add(pixel.r / 255.0)
    //     ..add(pixel.g / 255.0)
    //     ..add(pixel.b / 255.0);
    // }

    return Float32List.fromList(normalizedValues);
  }

  Map<String, OrtValueTensor> _createInputs({required img.Image image, required Float32List preprocessedInput}) {
    final input1TensorShape = [1, 3, 416, 416];
    final input1Tensor = OrtValueTensor.createTensorWithDataList(preprocessedInput, input1TensorShape);

    final imageShape = Float32List.fromList([image.height.toDouble(), image.width.toDouble()]);
    final imageShapeTensorShape = [1, 2];
    final imageShapeTensor = OrtValueTensor.createTensorWithDataList(imageShape, imageShapeTensorShape);

    final inputs = {'input_1': input1Tensor, 'image_shape': imageShapeTensor};
    return inputs;
  }
}

class OutputBoxes {
  OutputBoxes({required this.outputBoxes, required this.originalWidth, required this.originalHeight});

  static OutputBoxes? from(List<OrtValue?>? outputs, {required int originalWidth, required int originalHeight}) {
    // Checking if the outputs are there and if there are exactly 3 of them
    if (outputs == null || outputs.length != 3) {
      print('Incorrect outputs');
      return null;
    }

    // All boxes found by the model as 3D double array [1 x num_boxes x 4]
    final outputBoxes = outputs[0]?.value;
    if (outputBoxes == null || outputBoxes is! List<List<List<double>>>) {
      print('Incorrect boxes');
      return null;
    }

    // All scores for each class for each found box as 3D double array [1 x 80 x num_boxes]
    final outputScores = outputs[1]?.value;
    if (outputScores == null || outputScores is! List<List<List<double>>>) {
      print('Incorrect scores');
      return null;
    }

    // Indices of the best results as a 3D int array [1 x num_results x 3].
    // Each result is an array containing 3 indices as follows: [sample_index, class_index, box_index]
    final outputIndices = outputs[2]?.value;
    if (outputIndices == null || outputIndices is! List<List<List<int>>>) {
      print('Incorrect indices');
      return null;
    }

    // // Iterating over the best results
    // final List<OutputBox> results = [];
    // for (final index in outputIndices[0]) {
    //   final classIndex = index[1];
    //   final boxIndex = index[2];
    //
    //   // We're only looking for people on the photo, which has classIndex of 0
    //   if (classIndex != 0) {
    //     continue;
    //   }
    //
    //   // Finding the box
    //   print('Box found: ${outputBoxes[0][boxIndex]}');
    //   final box = outputBoxes[0][boxIndex];
    //   results.add(OutputBox.from(box));
    // }

    // Iterating over the best results
    final List<OutputBox> results = [];
    final personScores = outputScores[0][0];
    for (int scoreIndex = 0; scoreIndex < personScores.length; ++scoreIndex) {
      const threshold = 0.1;
      final score = personScores[scoreIndex];
      if (score > threshold) {
        final box = outputBoxes[0][scoreIndex];
        results.add(OutputBox.from(box));
      }
    }

    return OutputBoxes(outputBoxes: results, originalWidth: originalWidth, originalHeight: originalHeight);
  }

  final List<OutputBox> outputBoxes;
  final int originalWidth;
  final int originalHeight;
}

class OutputBox {
  OutputBox({required this.width, required this.height, required this.centerX, required this.centerY});

  final int width;
  final int height;
  final int centerX;
  final int centerY;

  factory OutputBox.from(List<double> box) {
    return OutputBox(
      width: box[0].toInt(),
      height: box[1].toInt(),
      centerX: box[2].toInt(),
      centerY: box[3].toInt(),
      // width: box[0].toInt(),
      // height: box[2].toInt(),
      // centerX: box[1].toInt(),
      // centerY: box[3].toInt(),
    );
  }
}
