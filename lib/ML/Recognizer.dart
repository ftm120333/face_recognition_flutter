import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../DB/DatabaseHelper.dart';
import 'Recognition.dart';

class Recognizer {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;
  static const int WIDTH = 112;
  static const int HEIGHT = 112;
  final dbHelper = DatabaseHelper();
  Map<String, Recognition> registered = Map();
  @override
  String get modelName => 'assets/mobile_face_net.tflite';

  Recognizer({int? numThreads}) {
    _interpreterOptions = InterpreterOptions();

    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
    loadModel();
    initDB();
  }

  initDB() async {
    await dbHelper.init();
    loadRegisteredFaces();
  }

  void loadRegisteredFaces() async {
    registered.clear();
    final allRows = await dbHelper.queryAllRows();
    // debugPrint('query all rows:');
    for (final row in allRows) {
      //  debugPrint(row.toString());
      print(row[DatabaseHelper.columnName]);
      String name = row[DatabaseHelper.columnName];
      List<double> embd = row[DatabaseHelper.columnEmbedding]
          .split(',')
          .map((e) => double.parse(e))
          .toList()
          .cast<double>();
      Recognition recognition =
          Recognition(row[DatabaseHelper.columnName], Rect.zero, embd, 0);
      registered.putIfAbsent(name, () => recognition);
      print("R=" + name);
    }
  }

  // void registerFaceInDB(String name, List<double> embedding) async {
  //   // row to insert
  //   Map<String, dynamic> row = {
  //     DatabaseHelper.columnName: name,
  //     DatabaseHelper.columnEmbedding: embedding.join(",")
  //   };
  //   final id = await dbHelper.insert(row);
  //   print('inserted row id: $id');
  //   loadRegisteredFaces();
  // }

//changes for multiple snapshots per person
  Future<void> registerFaceInDB(String name, List<double> embedding) async {
    //check if face already exists
    if (registered.containsKey(name)) {
      List<double> existing = registered[name]!.embeddings;
      List<double> averaged = List<double>.from(existing);

      for (int i = 0; i < embedding.length; i++) {
        averaged[i] = (existing[i] * registered[name]!.samples + embedding[i]) /
            (registered[name]!.samples + 1);
      }

      Map<String, dynamic> row = {
        DatabaseHelper.columnName: name,
        DatabaseHelper.columnEmbedding: averaged.join(",")
      };
      await dbHelper.update(row);
    } else {
      // New registration
      Map<String, dynamic> row = {
        DatabaseHelper.columnName: name,
        DatabaseHelper.columnEmbedding: embedding.join(",")
      };
      await dbHelper.insert(row);
    }
    loadRegisteredFaces();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset(modelName);
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  List<dynamic> imageToArray(img.Image inputImage) {
    img.Image resizedImage =
        img.copyResize(inputImage!, width: WIDTH, height: HEIGHT);
    List<double> flattenedList = resizedImage.data!
        .expand((channel) => [channel.r, channel.g, channel.b])
        .map((value) => value.toDouble())
        .toList();
    Float32List float32Array = Float32List.fromList(flattenedList);
    int channels = 3;
    int height = HEIGHT;
    int width = WIDTH;
    Float32List reshapedArray = Float32List(1 * height * width * channels);
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index = c * height * width + h * width + w;
          reshapedArray[index] =
              (float32Array[c * height * width + h * width + w] - 127.5) /
                  127.5;
        }
      }
    }
    return reshapedArray.reshape([1, 112, 112, 3]);
  }

  Recognition recognize(img.Image image, Rect location) {
    //TODO crop face from image resize it and convert it to float array
    var input = imageToArray(image);
    print(input.shape.toString());

    //TODO output array
    List output = List.filled(1 * 192, 0).reshape([1, 192]);

    //TODO performs inference
    final runs = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(input, output);
    final run = DateTime.now().millisecondsSinceEpoch - runs;
    print('Time to run inference: $run ms$output');

    //TODO convert dynamic list to double list
    List<double> outputArray = output.first.cast<double>();

    //TODO looks for the nearest embeeding in the database and returns the pair
    Pair pair = findNearest(outputArray);
    print("distance= ${pair.distance}");

    return Recognition(pair.name, location, outputArray, pair.distance);
  }

  //TODO  looks for the nearest embeeding in the database and returns the pair which contain information of registered face with which face is most similar
  // findNearest(List<double> emb) {
  //   Pair pair = Pair("Unknown", -5);
  //   for (MapEntry<String, Recognition> item in registered.entries) {
  //     final String name = item.key;
  //     List<double> knownEmb = item.value.embeddings;
  //     double distance = 0;
  //     for (int i = 0; i < emb.length; i++) {
  //       double diff = emb[i] - knownEmb[i];
  //       distance += diff * diff;
  //     }
  //     distance = sqrt(distance);
  //     if (pair.distance == -5 || distance < pair.distance) {
  //       pair.distance = distance;
  //       pair.name = name;
  //     }
  //   }
  //   return pair;
  // }

  Pair findNearest(List<double> emb) {
    Pair pair = Pair("Unknown", -5);

    for (MapEntry<String, Recognition> item in registered.entries) {
      final String name = item.key;
      List<double> knownEmb = item.value.embeddings;

      // Try both Euclidean and Cosine distance
      double distance = calculateCosineDistance(emb, knownEmb);

      print("Distance to $name: $distance");

      if (pair.distance == -5 || distance < pair.distance) {
        pair.distance = distance;
        pair.name = name;
      }
    }

    return pair;
  }

  double calculateCosineDistance(List<double> emb1, List<double> emb2) {
    double dotProduct = 0;
    double norm1 = 0;
    double norm2 = 0;

    for (int i = 0; i < emb1.length; i++) {
      dotProduct += emb1[i] * emb2[i];
      norm1 += emb1[i] * emb1[i];
      norm2 += emb2[i] * emb2[i];
    }

    double similarity = dotProduct / (sqrt(norm1) * sqrt(norm2));
    return 1 - similarity; // Convert to distance metric
  }

  void close() {
    interpreter.close();
  }
}

class Pair {
  String name;
  double distance;
  Pair(this.name, this.distance);
}
