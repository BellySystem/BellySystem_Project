#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <WiFi.h>
#include <OSCMessage.h>
#include <OSCBundle.h>
#include <WiFiUdp.h>

// === CONFIGURACIÓN ===
const char* ssid = "WIFI_NETWORK_NAME";
const char* password = "WIFI_NETWORK_PASSWORD";
const char* remoteIP = "XX.XXX.XX.XXX";   //IP ADRESS
const int remotePort = 8000;

// Configuración de muestreo
#define TARGET_SAMPLE_RATE 50    // Hz deseado
#define LOOP_PERIOD_MS 20        // 1000ms / 50Hz = 20ms
#define SERIAL_PRINT_INTERVAL 500 // Imprimir cada 500ms para no ralentizar

// Configuración del sensor
#define ACCEL_RANGE MPU6050_RANGE_8_G
#define GYRO_RANGE MPU6050_RANGE_1000_DEG
#define FILTER_BW MPU6050_BAND_44_HZ

Adafruit_MPU6050 mpu;
WiFiUDP udp;

// Mensajes OSC (creados una sola vez, reutilizados)
OSCBundle bundle;

// Variables de timing
unsigned long lastSampleTime = 0;
unsigned long lastPrintTime = 0;
unsigned long sampleCount = 0;
String cachedIP = "";  // IP cacheada

// Estadísticas de rendimiento
float avgLoopTime = 0;
float maxLoopTime = 0;

void setup() {
  // Serial solo para debug inicial (opcional)
  // Serial.begin(115200);
  
  // Inicializar I2C
  Wire.begin(21, 22);
  Wire.setClock(400000);  // I2C Fast Mode (400kHz)
  delay(100);

  // Inicializar MPU6050
  if (!mpu.begin(0x68, &Wire)) {
    // Si falla, parpadear LED (opcional)
    while (1) delay(10);
  }

  // Configurar rangos del sensor
  mpu.setAccelerometerRange(ACCEL_RANGE);
  mpu.setGyroRange(GYRO_RANGE);
  mpu.setFilterBandwidth(FILTER_BW);

  // Conectar WiFi
  setupWiFi();
  udp.begin(8888);
  
  lastSampleTime = millis();
  lastPrintTime = millis();
}

void loop() {
  unsigned long loopStartTime = millis();
  
  // ===== 1. TIMING PRECISO =====
  // Solo muestrear cuando hayan pasado exactamente 20ms
  if (loopStartTime - lastSampleTime < LOOP_PERIOD_MS) {
    return;  // Salir temprano si no es tiempo aún
  }
  
  unsigned long actualDelta = loopStartTime - lastSampleTime;
  lastSampleTime = loopStartTime;
  
  // ===== 2. LECTURA DEL SENSOR =====
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  // ===== 3. ENVÍO OSC OPTIMIZADO (Bundle = 1 solo paquete UDP) =====
  bundle.empty();  // Limpiar bundle anterior
  
  OSCMessage &msgAcc = bundle.add("/acc/xyz");
  msgAcc.add(a.acceleration.x);
  msgAcc.add(a.acceleration.y);
  msgAcc.add(a.acceleration.z);
  
  OSCMessage &msgGyr = bundle.add("/gyr/xyz");
  msgGyr.add(g.gyro.x);
  msgGyr.add(g.gyro.y);
  msgGyr.add(g.gyro.z);
  
  // Enviar todo en UN SOLO paquete UDP
  udp.beginPacket(remoteIP, remotePort);
  bundle.send(udp);
  udp.endPacket();
  
  sampleCount++;
  
  // ===== 4. RECONEXIÓN WiFi NO BLOQUEANTE =====
  if (sampleCount % 250 == 0) {  // Verificar cada 5 segundos
    if (WiFi.status() != WL_CONNECTED) {
      WiFi.reconnect();  // No bloqueante
      cachedIP = "";
    }
  }
 
}

void setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  // Esperar conexión (solo al inicio)
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 40) {
    delay(250);
    retry++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    cachedIP = WiFi.localIP().toString();
  }
}
