/*  
*  Copyright (c) 2018 Intel Corporation.
*  Licensed under the MIT license. See LICENSE file in the project root for full license information.
*/

/* 
  Hello World

  Sends various system commands to show network, user, BIOS, CPU, and OS information
  
  Make sure to open the Monitor before running.
  
  https://github.com/intel-iot-devkit/iei-tank-iot-developer-kit/tree/master/examples/HelloWorld
*/

void setup() {
   //Network settings
   system("ip a");
   //user name
   system("whoami");
   //BIOS info
   system("dmidecode -t bios");
   //CPU info
   system("lscpu");
   //Ubuntu info
   system("lsb_release -a");
}

  void loop() {
   printf("hello world");
   delay(5000);
}
