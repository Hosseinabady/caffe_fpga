#ifndef WIN32

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/ioctl.h>		/* ioctl */
#include <sys/mman.h>
#include <signal.h>
#include <fpgacl_device_driver.h>
#include "caffe/fpga_accel_driver.hpp"

#define DEVICE_NAME_WITH_PATH    "/dev/fpga_dev"




FILE *file;
FILE *fp;
int file_desc;

register_control_status_command_type *rd_command;

u32 device_kernel_address;


static void
accelerator_finished(int sig, siginfo_t *siginfo, void *context)		/* argument is signal number */
{
//	double tc = getTimestamp();
//	printf ("Sending PID: %ld, UID: %ld at %f\n", (long)siginfo->si_pid, (long)siginfo->si_uid, tc);
//	printf ("accelerator_finished:\n");
	return;
}

void accel_prologue() {
//	printf("accel_prologue ceck point 1\n");

	struct sigaction act;
	memset (&act, '\0', sizeof(act));
	act.sa_sigaction = &accelerator_finished;
	act.sa_flags = SA_SIGINFO;


	if (sigaction(SIGUSR1, &act, NULL) < 0)
		printf("can't catch SIGUSR1\n");


	file_desc = open(DEVICE_NAME_WITH_PATH, O_RDWR);
    if (file_desc < 0) {
        printf("Can't open device file :%s\n", DEVICE_NAME_WITH_PATH);
        exit(-1);
    }



    rd_command = (register_control_status_command_type*)malloc(sizeof(register_control_status_command_type));
    u32 process_id = getpid();

 //   printf("accel_prologue: process_id = %d\n", process_id);
 //   printf("accel_prologue: process_id = %d\n", (u32)rd_command);

	rd_command->device_physical_base_address = ACCELERATOR;
	rd_command->irq_num = 61;
	rd_command->process_id = process_id;
	device_kernel_address = ioctl (file_desc, FPGACL_REGISTER_DEVICE, rd_command);


    printf("accel_prologue ceck point 3\n");

}

void accel_epilogue() {
	printf("check point accel_epilogue 1\n");

	free(rd_command);

	close(file_desc);

}

void  accel_write(
				data_t  	*data_input_0,
				data_t  	*data_input_1,
				data_t  	*data_input_2,
				data_t  	*data_input_3,

				data_t      *data_output_in,
				data_t      *data_output_out,
				data_t      *weights,
				data_t      *weights_bias,

				u32          n,

				u32          param_input_h,
				u32          param_input_w,
				u32          param_input_c,
				u32          param_input_s,

				u32          param_output_h,
				u32          param_output_w,
				u32          param_output_c,
				u32          param_output_s,

				u32          output_p,

				u32          param_kernel_h,
				u32          param_kernel_w
		)
{



  	//=======================HP0-0 write command========================================================
  	//---------------------------------------
	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_DATA_INPUT_0_DATA;
	rd_command->value = (u32)(data_input_0);
	rd_command->pointer = 1;
	rd_command->direction=1;
	//rd_command->size_type = sizeof(unsigned char);
	rd_command->size_type = sizeof(u32);


	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_DATA_INPUT_1_DATA;
	rd_command->value = (u32)(data_input_1);
	rd_command->pointer = 1;
	rd_command->direction=1;
	//rd_command->size_type = sizeof(unsigned char);
	rd_command->size_type = sizeof(u32);


	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------


	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_DATA_INPUT_2_DATA;
	rd_command->value = (u32)(data_input_2);
	rd_command->pointer = 1;
	rd_command->direction=1;
	//rd_command->size_type = sizeof(unsigned char);
	rd_command->size_type = sizeof(u32);


	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------


	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_DATA_INPUT_3_DATA;
	rd_command->value = (u32)(data_input_3);
	rd_command->pointer = 1;
	rd_command->direction=1;
	//rd_command->size_type = sizeof(unsigned char);
	rd_command->size_type = sizeof(u32);


	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_DATA_OUTPUT_IN_DATA;
	rd_command->value = (u32)(data_output_in);
	rd_command->pointer = 1;
	rd_command->direction=3;
	rd_command->size_type = sizeof(u32);


	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------
	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_DATA_OUTPUT_OUT_DATA;
	rd_command->value = (u32)(data_output_out);
	rd_command->pointer = 1;
	rd_command->direction=3;
	rd_command->size_type = sizeof(u32);


	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------


	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_WEIGHTS_DATA;
	rd_command->value = (u32)(weights);
	rd_command->pointer = 1;
	rd_command->direction=1;
	rd_command->size_type = sizeof(u32);


	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_WEIGHTS_BIAS_DATA;
	rd_command->value = (u32)(weights_bias);
	rd_command->pointer = 1;
	rd_command->direction=1;
	rd_command->size_type = sizeof(u32);


	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

  	//---------------------------------------

	rd_command->device_kernel_base_address = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_N_V_DATA;
	rd_command->value = (u32)(n);
	rd_command->pointer = 0;
	rd_command->direction = 0;
	rd_command->size_type = sizeof(int);

	ioctl(file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
	//---------------------------------------
	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_INPUT_H_V_DATA;
	rd_command->value = (u32)(param_input_h);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(int);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_INPUT_W_V_DATA;
	rd_command->value = (u32)(param_input_w);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(int);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_INPUT_C_V_DATA;
	rd_command->value = (u32)(param_input_c);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(float);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------


	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_INPUT_S_V_DATA;
	rd_command->value = (u32)(param_input_s);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(float);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------



	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_OUTPUT_H_V_DATA;
	rd_command->value = (u32)(param_output_h);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(int);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_OUTPUT_W_V_DATA;
	rd_command->value = (u32)(param_output_w);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(int);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_OUTPUT_C_V_DATA;
	rd_command->value = (u32)(param_output_c);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(float);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------


	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_OUTPUT_S_V_DATA;
	rd_command->value = (u32)(param_output_s);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(int);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_OUTPUT_P_V_DATA;
	rd_command->value = (u32)(output_p);
	rd_command->pointer = 0;
	rd_command->direction = 0;
	rd_command->size_type = sizeof(int);

	ioctl(file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
	//---------------------------------------
	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_KERNEL_H_V_DATA;
	rd_command->value = (u32)(param_kernel_h);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(float);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_PARAM_KERNEL_W_V_DATA;
	rd_command->value = (u32)(param_kernel_w);
	rd_command->pointer = 0;
	rd_command->direction=0;
	rd_command->size_type = sizeof(float);

	ioctl ( file_desc, FPGACL_DEVICE_IOWRITE, rd_command);
  	//---------------------------------------
}




void  accel_start( ) {
//	printf("accel_start: ceck point 1\n");
	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_GIE;
	rd_command->value = 0;
  	ioctl ( file_desc, FPGACL_INTERRUPT_ENABLE, rd_command);

  	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_AP_CTRL;
	rd_command->value = 0;
	ioctl ( file_desc, FPGACL_START, rd_command);
}



void  accel_ready() {
	printf("accel_wait: ceck point 1\n");
	u32 isIdle = 0;
	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_AP_CTRL;
	rd_command->value = 0;
	isIdle  =  ioctl ( file_desc, FPGACL_CTRL_READY, rd_command);

	while (!isIdle) {
		printf("accel_wait: ceck point 2\n");
		sleep(1);
		isIdle  =   ioctl ( file_desc, FPGACL_CTRL_READY, rd_command);
	}

}


void  accel_wait() {
//	printf("accel_wait: ceck point 1\n");
	u32 isIdle = 0;
	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->register_physical_offset_address = XCNN_ACCEL_CONTROL_ADDR_AP_CTRL;
	rd_command->value = 0;
	isIdle  =  ioctl ( file_desc, FPGACL_CTRL_DONE, rd_command);

	while (!isIdle) {
		isIdle  =   ioctl ( file_desc, FPGACL_CTRL_DONE, rd_command);
	}

}

double getTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}

void* fpgacl_malloc(unsigned long int size) {
	return fpgacl_cacheable_malloc(size);
}

void* fpgacl_cacheable_malloc(unsigned long int size) {
	void* user_var;

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->value = (u32)(user_var);

	rd_command->mem_cacheable = 1;

	ioctl ( file_desc, FPGACL_MEMCACHEABLE, rd_command);

	user_var = mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED| MAP_LOCKED, file_desc, 0);
	if (user_var == MAP_FAILED)	{
		perror("file_desc:--");
		exit(-1);
	}

	return user_var;
}


void* fpgacl_noncacheable_malloc(unsigned long int size) {
	void* user_var;

	rd_command->device_kernel_base_address     = (void *)device_kernel_address;
	rd_command->value = (u32)(user_var);

	rd_command->mem_cacheable = 0;

	ioctl ( file_desc, FPGACL_MEMCACHEABLE, rd_command);

	user_var = mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED| MAP_LOCKED, file_desc, 0);
	if (user_var == MAP_FAILED)	{
		perror("file_desc:--");
		exit(-1);
	}



	return user_var;
}


void fpgacl_free(void* user_var) {

	ioctl ( file_desc, FPGACL_MEMFREE, user_var);
}



/*
void* fpgacl_malloc(unsigned long int size) {
	void* user_var;

	user_var = mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED| MAP_LOCKED, file_desc, 0);
	if (user_var == MAP_FAILED)	{
		perror("file_desc:--");
		exit(-1);
	}

	return user_var;
}

void fpgacl_free(void* user_var) {

	ioctl ( file_desc, FPGACL_MEMFREE, user_var);
}
*/
#else

#endif //WIN32

