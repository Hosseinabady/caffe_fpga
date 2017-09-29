#ifndef __FPGA_BASIC__H__
#define __FPGA_BASIC__H__

#include <stdint.h>

#define DATA_HEIGHT_MAX     512
#define DATA_WIDTH_MAX      512

#define KERNEL_HEIGHT_MAX   3
#define KERNEL_WIDTH_MAX    KERNEL_HEIGHT_MAX

#define BUFFER_HIGHT_MAX    KERNEL_HEIGHT_MAX
#define BUFFER_WIDTH_MAX    DATA_WIDTH_MAX+KERNEL_WIDTH_MAX



#define NO_IN_CHANNEL_MAX   512
#define NO_OUT_CHANNEL_MAX  512

typedef unsigned long int u32;
typedef float data_t;
typedef float weight_t;
typedef int16_t param_t;




struct parameter_s {
	//	u32 w; // width size
	//	u32 h; // hight size
	//	u32 c; // number of channel
	//	u32 s; //
	//	u32 p; // padding

	param_t w; // width size
	param_t h; // hight size
	param_t c; // number of channel
	param_t s; //
	param_t p; // padding
};
typedef parameter_s parameter_t;





#endif //__FPGA_BASIC__H__
