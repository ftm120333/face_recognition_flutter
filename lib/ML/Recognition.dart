import 'dart:ui';

class Recognition {
  String name;
  Rect location;
  List<double> embeddings;
  double distance;
  int samples; //Adding this to track number of samples
  /// Constructs a Category.
  Recognition(this.name, this.location, this.embeddings, this.distance,
      {this.samples = 1});
}
