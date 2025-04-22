#define MODEL_NAME TakuNet_Random_9
#define IMAGE_NAME image_9   
#define XSTR(x) STR(x)
#define STR(x) #x

#include XSTR(MODEL_NAME.h)     // Include the selected model
#include XSTR(IMAGE_NAME.h)     // Include the selected image

#include "cifar100_labels.h"

#define CAT(a, b) a##b
#define EXPAND_AND_CAT(a, b) CAT(a, b)
#define MODEL_DATA EXPAND_AND_CAT(MODEL_NAME, _data)
#define MODEL_DATA_LEN (sizeof(MODEL_DATA))

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"

#define GREEN_LED  LEDB
#define RED_LED   LEDR
#define BLUE_LED LEDG


constexpr size_t TOTAL_FLASH_BYTES = 1048576; // 1 MB total flash for Nano 33 BLE

// 💾 Memory estimation based on model: ~128 KB → + safety buffer
// Based on this number we allocate 143360 Bytes in the Tensor Arena
// By doing so we ensure that all the Models that are estimated at 128 KB will fit
// I can potentially increase it -> Increasing the actual size of the Model's Ram

constexpr int kTensorArenaSize = 140 * 1024;  // 140 KB arena
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


void blinkColor(int pin, int times, int onDuration = 500, int offDuration = 150) {
  for (int i = 0; i < times; i++) {
    digitalWrite(pin, LOW);
    delay(onDuration);
    digitalWrite(pin, HIGH);
    delay(offDuration);
  }
}

void setup() {
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);

  digitalWrite(RED_LED, HIGH);
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(BLUE_LED, HIGH);

  Serial.begin(115200);// Starts the USB serial connection at 115200 baud (standard speed for logging).
  while (!Serial);  // Wait for Serial Monitor, This must be commented out if I power Arduino with PPK2


  Serial.println("🚀 Initializing TFLite model...");

  //#define MODEL_DATA_LEN (sizeof(MODEL_NAME##_data))

  Serial.println("📦 Flash Memory Info");
  Serial.print("💾 Model Flash Size: ");
  Serial.print(MODEL_DATA_LEN);
  Serial.print(" bytes (");
  Serial.print(MODEL_DATA_LEN / 1024.0, 2);
  Serial.println(" KB)");

  Serial.print("📊 Model Flash Usage: ");
  Serial.print((MODEL_DATA_LEN * 100.0) / TOTAL_FLASH_BYTES, 2);
  Serial.println(" % of total flash (1 MB)");

  const tflite::Model* model = tflite::GetModel(MODEL_DATA); //Parses the .h file array TakuNet_Random_9_data into a TFLite model object.
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
  resolver.AddSub();
  resolver.AddMul();
  resolver.AddRsqrt();
  resolver.AddRelu6(); 





  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize); // Allocates memory and wires up the model.
  interpreter = &static_interpreter; // A preallocated chunk of RAM for all tensors (from inputs to outputs)

  Serial.println("📦 Allocating tensors...");
  unsigned long alloc_start = micros();  

  if (interpreter->AllocateTensors() != kTfLiteOk) { // Allocates input/output/intermediate buffers inside tensor_aren
    Serial.println("❌ Failed to allocate tensors!"); //  Fails if the arena is too small.
    return;
  }

  
  unsigned long alloc_time  = micros() - alloc_start;  // Calculate elapsed time
  Serial.print("⏱️ Allocating Tensors Time: ");
  Serial.print(alloc_time);
  Serial.print(" µs (");
  Serial.print(alloc_time / 1000000.0, 4); // convert to seconds
  Serial.println(" s)");
  

  /* These pointers let you read/write data to/from the model.
  CIFAR-100 uses one input tensor of 3072 bytes (32x32x3 RGB image).
  Output is typically a 100-class probability distribution.
  */

  input = interpreter->input(0);
  output = interpreter->output(0);

  // 🔍 Show actual memory usage
  size_t used_bytes = interpreter->arena_used_bytes();
  
  Serial.print("📦 Max RAM needed (arena used): ");
  Serial.print(used_bytes);
  Serial.print(" bytes (");
  Serial.print(used_bytes / 1024.0, 2);  // print in KB with 2 decimal places
  Serial.println(" KB)");
  

  if (input->bytes != 3072) {
    //Serial.print("❌ Input size mismatch. Expected 3072, got ");
    //Serial.println(input->bytes);
    return;
  }

  /* Load image into input tensor
   Copy Image into Input Tensor
  */

  for (int i = 0; i < 3072; i++) {
    input->data.uint8[i] = IMAGE_NAME[i];
  }

  // Run inference
  Serial.println("🔮 Running inference...");
  unsigned long infer_start = micros();
  if (interpreter->Invoke() != kTfLiteOk) { //Performs a forward pass through the model using the input data.
    Serial.println("❌ Inference failed!");
    return;
  }
  unsigned long infer_time = micros() - infer_start;
  Serial.print("⏱️ Actual Inference Time (µs): ");
  Serial.print(infer_time);
  Serial.print(" µs (");
  Serial.print(infer_time / 1000000.0, 4); // convert to seconds
  Serial.println(" s)");

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

    // ✅ Blink LED once to signal inference is done
  digitalWrite(LED_BUILTIN, HIGH);
  delay(200);
  digitalWrite(LED_BUILTIN, LOW);

  delay(500);
  blinkColor(RED_LED, top_class); // short pause before blinking, also choose your Color Here
  
  Serial.print("✅ Predicted Class Index: ");
  Serial.println(top_class);
  Serial.print("🧠 Predicted Label: ");
  Serial.println(CIFAR100_LABELS[top_class]);

  float scale = output->params.scale;
  int zero_point = output->params.zero_point;
  float probability = (static_cast<int>(max_val) - zero_point) * scale;
  Serial.print("🎯 Confidence (float): ");
  Serial.println(probability, 4);

  Serial.print("Probability (uint8): ");
  Serial.println(max_val);
}

void loop() {
  // Nothing in loop — one-time inference in setup
}
