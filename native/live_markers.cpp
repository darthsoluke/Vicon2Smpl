// Read-only Linux DataStream bridge. Build against the separately installed official SDK.
#include "DataStreamClient.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
using namespace ViconDataStreamSDK::CPP;

void quoted(const std::string &s) {
  std::cout << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') std::cout << '\\' << c;
    else if (c < 32) std::cout << '?';
    else std::cout << c;
  }
  std::cout << '"';
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  Client client;
  client.SetConnectionTimeout(3000);
  if (client.Connect(argv[1]).Result != Result::Success) return 3;
  if (client.EnableMarkerData().Result != Result::Success) return 4;
  // Explicit right-handed Forward/Left/Up axes; the Python adapter converts mm/Z-up to m/Y-up.
  if (client.SetAxisMapping(Direction::Forward, Direction::Left, Direction::Up).Result != Result::Success) return 5;
  if (client.SetStreamMode(StreamMode::ServerPush).Result != Result::Success) return 6;
  client.SetBufferSize(1);
  std::cout << std::setprecision(12);
  while (client.IsConnected().Connected) {
    if (client.GetFrame().Result != Result::Success) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    auto count = client.GetSubjectCount();
    if (count.Result != Result::Success) continue;
    std::cout << "{\"frame\":" << client.GetFrameNumber().FrameNumber
              << ",\"fps\":" << client.GetFrameRate().FrameRateHz << ",\"subjects\":[";
    for (unsigned int i = 0; i < count.SubjectCount; ++i) {
      if (i) std::cout << ',';
      auto name = client.GetSubjectName(i).SubjectName;
      std::cout << "{\"name\":"; quoted(name); std::cout << ",\"markers\":[";
      auto markers = client.GetMarkerCount(name);
      for (unsigned int j = 0; j < markers.MarkerCount; ++j) {
        if (j) std::cout << ',';
        auto label = client.GetMarkerName(name, j).MarkerName;
        auto p = client.GetMarkerGlobalTranslation(name, label);
        bool valid = p.Result == Result::Success && !p.Occluded;
        for (double x : p.Translation) valid = valid && std::isfinite(x);
        std::cout << "{\"name\":"; quoted(label);
        std::cout << ",\"valid\":" << (valid ? "true" : "false") << ",\"xyz\":";
        if (valid) std::cout << '[' << p.Translation[0] << ',' << p.Translation[1] << ',' << p.Translation[2] << ']';
        else std::cout << "null";
        std::cout << '}';
      }
      std::cout << "]}";
    }
    std::cout << "]}\n" << std::flush;
  }
  client.Disconnect();
}
