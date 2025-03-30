#include "TakuNet_Random_9.h"  // Your model header
#include "image_9.h"           // Your image data
#include "cifar100_labels.h"


#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"

// 💾 Memory estimation based on model: ~128 KB → + safety buffer
constexpr int kTensorArenaSize = 140 * 1024;  // 140 KB arena
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);// Starts the USB serial connection at 115200 baud (standard speed for logging).
  while (!Serial);  // Wait for Serial Monitor

  Serial.println("🚀 Initializing TFLite model...");

  const tflite::Model* model = tflite::GetModel(TakuNet_Random_9_data); //Parses the .h file array TakuNet_Random_9_data into a TFLite model object.
  if (model->version() != TFLITE_SCHEMA_VERSION) { //Checks if the model version is compatible (should always be TFLITE_SCHEMA_VERSION).
    Serial.println("❌ Incompatible TFLite model schema.");
    return;
  }

  static tflite::MicroMutableOpResolver<17> resolver;


  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddAdd();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddConcatenation();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddNeg();
  resolver.AddSquaredDifference();
  resolver.AddSub();        // <-- for LayerNorm
  resolver.AddMul();        // <-- for LayerNorm
  resolver.AddRsqrt();      // <-- for LayerNorm
  resolver.AddRelu6(); 





  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize); // Allocates memory and wires up the model.
  interpreter = &static_interpreter; // A preallocated chunk of RAM for all tensors (from inputs to outputs)

  if (interpreter->AllocateTensors() != kTfLiteOk) { // Allocates input/output/intermediate buffers inside tensor_aren
    Serial.println("❌ Failed to allocate tensors!"); //  Fails if the arena is too small.
    return;
  }

  /* These pointers let you read/write data to/from the model.
  CIFAR-100 uses one input tensor of 3072 bytes (32x32x3 RGB image).
  Output is typically a 100-class probability distribution.
  */

  input = interpreter->input(0);
  output = interpreter->output(0);

  // 🔍 Show actual memory usage
  Serial.print("📦 Max RAM needed (arena used): ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" bytes");

  // Sanity check for input size
  if (input->bytes != 3072) {
    Serial.print("❌ Input size mismatch. Expected 3072, got ");
    Serial.println(input->bytes);
    return;
  }

  /* Load image into input tensor
   Copy Image into Input Tensor
  */

  for (int i = 0; i < 3072; i++) {
    input->data.uint8[i] = image_9[i];
  }

  // Run inference
  Serial.println("🔮 Running inference...");
  if (interpreter->Invoke() != kTfLiteOk) { //Performs a forward pass through the model using the input data.
    Serial.println("❌ Inference failed!");
    return;
  }

  // Find the predicted class (argmax)
  int top_class = -1;
  uint8_t max_val = 0;
  for (int i = 0; i < output->dims->data[1]; i++) {
    uint8_t val = output->data.uint8[i];
    if (val > max_val) {
      max_val = val;
      top_class = i;
    }
  }

  Serial.print("✅ Predicted Class Index: ");
  Serial.println(top_class);
  Serial.print("🧠 Predicted Label: ");
  Serial.println(CIFAR100_LABELS[top_class]);
  Serial.print("Probability (uint8): ");
  Serial.println(max_val);
}

void loop() {
  // Nothing in loop — one-time inference in setup
}
