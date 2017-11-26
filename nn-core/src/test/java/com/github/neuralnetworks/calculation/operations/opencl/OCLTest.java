package com.github.neuralnetworks.calculation.operations.opencl;

import com.github.neuralnetworks.calculation.operations.opencl.OCL;
import com.github.neuralnetworks.test.AbstractTest;

/**
 * @author Tony
 * @author Vadim
 */
public class OCLTest extends AbstractTest
{

	public void testMain() throws Exception
	{
//        System.load("d:\\classes\\ExBJCL.dll");
		OCL GPU = new OCL();

		int WeightedSum_miniBatchSize = 1;

		int WeightedSum_inputColumnStep = 1;
		int WeightedSum_inputRowStep = 2;
		int WeightedSum_inputStartPosition = 0;

		int WeightedSum_weightsInitialStep = 1;
		int WeightedSum_weightsSize = 3;
		int WeightedSum_weightsStep = 2;
		int WeightedSum_weightsStartPosition = 0;

		int WeightedSum_outputColumnStep = 1;
		int WeightedSum_outputRowStep = 3;
		int WeightedSum_outputStartPosition = 0;


		int Sigmoid_startIndex = 0;
		int Tanh_startIndex = 0;
		int ReLU_startIndex = 0;
		int SoftReLU_startIndex = 0;

		int ReLUDerivative_activationsStartIndex = 0;
		int ReLUDerivative_resultStartIndex = 0;
		int TanhDerivative_activationsStartIndex = 0;
		int TanhDerivative_resultStartIndex = 0;
		int SigmoidDerivative_activationsStartIndex = 0;
		int SigmoidDerivative_resultStartIndex = 0;
		int SoftReLUDerivative_activationsStartIndex = 0;
		int SoftReLUDerivative_resultStartIndex = 0;

		int Noise_startIndex = 0;
		float Noise_corruptionLevel = 0.1f;
		float Noise_corruptedValue = 0.2f;


		int[] dimensions = { WeightedSum_inputStartPosition + WeightedSum_inputRowStep * WeightedSum_miniBatchSize };
		float[] input = new float[32]; // sizes * miniBatchSize];
		float[] output1 = new float[400]; // sizes * miniBatchSize];
		float[] output2 = new float[400]; // sizes * miniBatchSize];
		float[] output3 = new float[400]; // sizes * miniBatchSize];
		float[] activations = new float[400]; // sizes * miniBatchSize];
		float[] weights = new float[4 * 4]; // weightsSize * weightsSize];

		int[] dimensions2 = { 4, 4 }; // { weightsSize, weightsSize };

//		input[0] = 0;		input[1] = 0;		input[2] = 0;		input[3] = 0;
//		input[4] = 1.0f;	input[5] = 4.0f;	input[6] = 2.0f;	input[7] = 5.0f;
//		input[8] = 3.0f;	input[9] = 6.0f;

//		output1[0] = 0;		output1[1] = 0;		output1[2] = 0;		output1[3] = 0;
//		output1[4] = 1.0f;	output1[5] = 4.0f;	output1[6] = 2.0f;	output1[7] = 5.0f;
//		output1[8] = 3.0f;	output1[9] = 6.0f;

//		output2[0] = 0;		output2[1] = 0;		output2[2] = 0;		output2[3] = 0;
//		output2[4] = 1.0f;	output2[5] = 4.0f;	output2[6] = 2.0f;	output2[7] = 5.0f;
//		output2[8] = 3.0f;	output2[9] = 6.0f;

//		output3[0] = 0;		output3[1] = 0;		output3[2] = 0;		output3[3] = 0;
//		output3[4] = 1.0f;	output3[5] = 4.0f;	output3[6] = 2.0f;	output3[7] = 5.0f;
//		output3[8] = 3.0f;	output3[9] = 6.0f;

//		weights[0] = 1.0f;	weights[1] = 4.0f;	weights[2] = 2.0f;	weights[3] = 5.0f;
//		weights[4] = 3.0f;	weights[5] = 6.0f;	weights[6] = 1.0f;	weights[7] = 4.0f;
//		weights[8] = 2.0f;	weights[9] = 5.0f;	weights[10]= 3.0f;	weights[11]= 6.0f;


		for (int i = 0; i < input.length; i++)
		{
			input[i] = 0;
		}

		for (int i = 0; i < WeightedSum_inputRowStep; i++)
		{
			input[i] = 2.0f;
		}

		for (int i = 0; i < weights.length; i++)
		{
			weights[i] = 0.2f;
		}

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0;
			output2[i] = 0;
			output3[i] = 0;
		}


		String opts =
				"";
		// weightedSum options
//					  " -D WSinSP="   + WeightedSum_inputStartPosition +
//					  " -D WSinRS="   + WeightedSum_inputRowStep +
//					  " -D WSinCS="   + WeightedSum_inputColumnStep +
//					  " -D WSwSP="    + WeightedSum_weightsStartPosition +
//					  " -D WSwIS="    + WeightedSum_weightsInitialStep +
//					  " -D WSwS="     + WeightedSum_weightsStep +
//				      " -D WSwZ="     + WeightedSum_weightsSize +
//				      " -D WSoutSP="  + WeightedSum_outputStartPosition +
//				      " -D WSoutRS="  + WeightedSum_outputRowStep +
//				      " -D WSoutCS="  + WeightedSum_outputColumnStep +
//				      " -D WSminiBS=" + WeightedSum_miniBatchSize
		// Sigmoid options
//				     +" -D SigSI="+Sigmoid_startIndex
		// Tanh options
//				     +" -D TanhSI="+Tanh_startIndex
		// ReLU options
//				     +" -D ReLUSI="+ReLU_startIndex
		// SoftReLU options
//				     +" -D SoftReLUSI="+SoftReLU_startIndex
		// ReLUDerivative options
//				     +" -D DReLUaSI="+ReLUDerivative_activationsStartIndex
//				     +" -D DReLUrSI="+ReLUDerivative_resultStartIndex
		// TanhDerivative options
//				     +" -D DTanhaSI="+TanhDerivative_activationsStartIndex
//				     +" -D DTanhrSI="+TanhDerivative_resultStartIndex
		// SigmoidDerivative options
//				     +" -D DSigaSI="+SigmoidDerivative_activationsStartIndex
//				     +" -D DSigrSI="+SigmoidDerivative_resultStartIndex
		// SoftReLUDerivative options
//				     +" -D DSReLUaSI="+SoftReLUDerivative_activationsStartIndex
//				     +" -D DSReLUrSI="+SoftReLUDerivative_resultStartIndex
		// Noise options
//				     +" -D NoiseCL="+Noise_corruptionLevel
//				     +" -D NoiseCV="+Noise_corruptedValue
//				     +" -D NoiseSI="+Noise_startIndex
//	     			 								;

//		GPU.setOptionsMode();

//        checkError( GPU.pathSetToCL( "d:\\classes".toCharArray() ) );

		char[] options = opts.toCharArray();

//		System.out.println("Devices: " + GPU.getDevicesNumber() );

		int deviceID1 = GPU.getDeviceID(0); // init device1
		checkError(GPU.initDeviceID(deviceID1, options, true));

		int deviceID2 = GPU.getDeviceID(deviceID1); // init device2
		checkError(GPU.initDeviceID(deviceID2, options, true));


		/*
		 * ---------------------
		 * weightedSum section
		 * ---------------------
		 */

		/* prepare buffers */
		int iID1 = GPU.prepareFloatArray(deviceID1, input, 0);
		int wID1 = GPU.prepareFloatArray(deviceID1, weights, 0);
		int oID1 = GPU.prepareFloatArray(deviceID1, output1, 0);

		int iID2 = GPU.prepareFloatArray(deviceID2, input, 0);
		int wID2 = GPU.prepareFloatArray(deviceID2, weights, 0);
		int oID2 = GPU.prepareFloatArray(deviceID2, output2, 0);


		/* prepare arrays */
		int inputID1 = GPU.prepareBuf(iID1, dimensions);
		int weightsID1 = GPU.prepareBuf(wID1, dimensions2);
		int outputID1 = GPU.prepareBuf(oID1, dimensions);

		int inputID2 = GPU.prepareBuf(iID2, dimensions);
		int weightsID2 = GPU.prepareBuf(wID2, dimensions2);
		int outputID2 = GPU.prepareBuf(oID2, dimensions);

		// int inputID1c = GPU.cloneFloatBuf(inputID1, deviceID1);
		// checkError( GPU.copyFloatBuf( inputID2, inputID1 ) );

		// checkError( GPU.logMessage( "Test message".toCharArray() ) );

		/* prepare kernels */
//		int kernelID1 = GPU.weightedSum( deviceID1, inputID1, weightsID1, outputID1, WeightedSum_inputRowStep );
//		int kernelID2 = GPU.weightedSum( deviceID2, inputID2, weightsID2, outputID2, WeightedSum_inputRowStep );

//		int kernelID1 = GPU.weightedSumN( deviceID1, inputID1, weightsID1, outputID1, WeightedSum_inputRowStep,
//		5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 );

		int kernelID1 = GPU.weightedSum(deviceID1, inputID1, weightsID1, outputID1, 3,
				WeightedSum_miniBatchSize, WeightedSum_inputStartPosition, WeightedSum_inputRowStep,
				WeightedSum_inputColumnStep, WeightedSum_outputStartPosition, WeightedSum_outputRowStep,
				WeightedSum_outputColumnStep, WeightedSum_weightsStartPosition, WeightedSum_weightsSize,
				WeightedSum_weightsInitialStep, WeightedSum_weightsStep, false);

		int kernelID2 = GPU.weightedSum(deviceID2, inputID2, weightsID2, outputID2, 3,
				WeightedSum_miniBatchSize, WeightedSum_inputStartPosition, WeightedSum_inputRowStep,
				WeightedSum_inputColumnStep, WeightedSum_outputStartPosition, WeightedSum_outputRowStep,
				WeightedSum_outputColumnStep, WeightedSum_weightsStartPosition, WeightedSum_weightsSize,
				WeightedSum_weightsInitialStep, WeightedSum_weightsStep, false);


		/* run kernels */
		String job = kernelID1 + "  " + kernelID2 + // async start 2 kernels
				" !" + kernelID1 + " !" + kernelID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices


		// checkError( GPU.kernelRunAsync( kernelID1 ) ); // push job to device1
		// checkError( GPU.kernelRunAsync( kernelID2 ) ); // push job to device2

		/* waiting for kernels' finishing */
		// do{} while ( ( GPU.kernelState( kernelID1 ) != 0 ) || ( GPU.kernelState( kernelID2 ) != 0 ));

		/* get results */
		checkError(GPU.getFloatBuf(outputID1, output1));
		checkError(GPU.getFloatBuf(outputID2, output2));


		System.out.println("");

		int indx = 1;
//		int id   = 0;
		float zr = 0;
		for (int id = 0; id < WeightedSum_outputRowStep; id++)
		{
			zr = 0;
			int inputStartIndex = 0;
			int outputStartIndex = WeightedSum_outputStartPosition + id * WeightedSum_outputColumnStep;
			int weightStartIndex = WeightedSum_weightsStartPosition + id * WeightedSum_weightsInitialStep;
			for (int i = 0; i < WeightedSum_miniBatchSize; i++)
			{
				// each connection (of the combined connections)
				zr = output3[outputStartIndex + i * WeightedSum_outputRowStep];
				// each element in the row/column

				inputStartIndex = WeightedSum_inputStartPosition + i * WeightedSum_inputRowStep;
				for (int j = 0; j < WeightedSum_weightsSize; j++)
				{
					zr += input[inputStartIndex + j * WeightedSum_inputColumnStep] *
							weights[weightStartIndex + j * WeightedSum_weightsStep];
				}

				output3[outputStartIndex + i * WeightedSum_outputRowStep] = zr;
			}
		}
		int id = 0;

		for (int i = 0; i < output1.length; i++)
		{
			System.out.print(output1[i] + " ");
		}
		System.out.println("");

		System.out.print("weightedSum         result[" + indx + "] device1: " + output1[indx] +
				"          device2: " + output2[indx] + "         CPU: " + output3[indx] +
				"         ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
		{
			System.out.println("[CORRECT!]");
		}
		else
		{
			System.out.println("[ERROR!]");
		}


		/*
		 * ---------------------
		 * Sigmoid section
		 * ---------------------
		 */

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.5f;
			output2[i] = 0.5f;
			output3[i] = 0.5f;
		}


		/* prepare buffers */
		int ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		int ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);

		/* prepare arrays */
//		int ioaID1   = GPU.prepareBuf( ioID1, dimensions  );
//		int ioaID2   = GPU.prepareBuf( ioID2, dimensions  );

		int ioaID1 = ioID1;
		int ioaID2 = ioID2;


		/* prepare kernels */
		// int SigmoidID1 = GPU.Sigmoid(deviceID1, ioaID1, 10);
		// int SigmoidID2 = GPU.Sigmoid(deviceID2, ioaID2, 10);

		int SigmoidID1 = GPU.Sigmoid(deviceID1, ioaID1, 10, Sigmoid_startIndex);
		int SigmoidID2 = GPU.Sigmoid(deviceID2, ioaID2, 10, Sigmoid_startIndex);


		job = SigmoidID1 + "  " + SigmoidID2 + // async start 2 kernels
				" !" + SigmoidID1 + " !" + SigmoidID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));


		for (int i = Sigmoid_startIndex; i < output3.length; i++)
		{
			output3[i] = (1 / (1 + (float) Math.exp(-output3[i])));
		}

		System.out.print("Sigmod              result[" + indx + "] device1: " + output1[indx] +
				"   device2: " + output2[indx] + "  CPU: " + output3[indx] + "  ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * Tanh section
		 * ---------------------
		 */

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.5f;
			output2[i] = 0.5f;
			output3[i] = 0.5f;
		}


		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);

		/* prepare kernels */
		// int tanhID1 = GPU.Tanh(deviceID1, ioaID1, 10);
		// int tanhID2 = GPU.Tanh(deviceID2, ioaID2, 10);

		int tanhID1 = GPU.Tanh(deviceID1, ioaID1, 10, Tanh_startIndex);
		int tanhID2 = GPU.Tanh(deviceID2, ioaID2, 10, Tanh_startIndex);

		job = tanhID1 + "  " + tanhID2 + // async start 2 kernels
				" !" + tanhID1 + " !" + tanhID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));

		for (int i = Tanh_startIndex; i < output3.length; i++)
		{
			output3[i] = (float) Math.tan(output3[i]); /// Why Math.tan?
		}

		System.out.print("Tanh                result[" + indx + "] device1: " + output1[indx] +
				"    device2: " + output2[indx] + "   CPU: " + output3[indx] + "   ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * ReLU section
		 * ---------------------
		 */

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = -0.5f;
			output2[i] = -0.5f;
			output3[i] = -0.5f;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);

		/* prepare kernels */
		// int ReLUID1 = GPU.ReLU(deviceID1, ioaID1, 10);
		// int ReLUID2 = GPU.ReLU(deviceID2, ioaID2, 10);

		int ReLUID1 = GPU.ReLU(deviceID1, ioaID1, 10, ReLU_startIndex);
		int ReLUID2 = GPU.ReLU(deviceID2, ioaID2, 10, ReLU_startIndex);

		job = ReLUID1 + "  " + ReLUID2 + // async start 2 kernels
				" !" + ReLUID1 + " !" + ReLUID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));

		for (int i = ReLU_startIndex; i < output3.length; i++)
		{
			output3[i] = Math.max(0, output3[i]);
		}

		System.out.print("ReLU                result[" + indx + "] device1: " + output1[indx] +
				"          device2: " + output2[indx] + "         CPU: " + output3[indx] +
				"         ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * SoftReLU section
		 * ---------------------
		 */

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.5f;
			output2[i] = 0.5f;
			output3[i] = 0.5f;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);

		/* prepare kernels */
		// int SoftReLUID1 = GPU.SoftReLU(deviceID1, ioaID1, 10);
		// int SoftReLUID2 = GPU.SoftReLU(deviceID2, ioaID2, 10);

		int SoftReLUID1 = GPU.SoftReLU(deviceID1, ioaID1, output1.length, SoftReLU_startIndex);
		int SoftReLUID2 = GPU.SoftReLU(deviceID2, ioaID2, output2.length, SoftReLU_startIndex);

		job = SoftReLUID1 + "  " + SoftReLUID2 + // async start 2 kernels
				" !" + SoftReLUID1 + " !" + SoftReLUID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));

		for (int i = SoftReLU_startIndex; i < output3.length; i++)
		{
			output3[i] = (float) Math.log(1 + Math.exp(output3[i]));
		}

		System.out.print("SoftReLU            result[" + indx + "] device1: " + output1[indx] +
				"     device2: " + output2[indx] + "    CPU: " + output3[indx] + "    ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * ReLUDerivative section
		 * ---------------------
		 */

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.5f;
			output2[i] = 0.5f;
			output3[i] = 0.5f;
			activations[i] = 0.5f;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);
		int acID1 = GPU.prepareFloatArray(deviceID1, activations, 0);
		int acID2 = GPU.prepareFloatArray(deviceID2, activations, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);
		int actID1 = GPU.prepareBuf(acID1, dimensions);
		int actID2 = GPU.prepareBuf(acID2, dimensions);

		/* prepare kernels */
		// int DReLUID1 = GPU.ReLUDerivative(deviceID1, ioaID1, actID1, 10);
		// int DReLUID2 = GPU.ReLUDerivative(deviceID2, ioaID2, actID2, 10);

		int DReLUID1 = GPU.ReLUDerivative(deviceID1, ioaID1, actID1, 10,
				ReLUDerivative_activationsStartIndex, ReLUDerivative_resultStartIndex);
		int DReLUID2 = GPU.ReLUDerivative(deviceID2, ioaID2, actID2, 10,
				ReLUDerivative_activationsStartIndex, ReLUDerivative_resultStartIndex);

		job = DReLUID1 + "  " + DReLUID2 + // async start 2 kernels
				" !" + DReLUID1 + " !" + DReLUID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));

		for (int i = ReLUDerivative_resultStartIndex; i < output3.length; i++)
		{
			if (activations[ReLUDerivative_activationsStartIndex] <= 0)
			{
				output3[i] = 0;
			}
		}

		System.out.print("ReLUDerivative      result[" + indx + "] device1: " + output1[indx] +
				"          device2: " + output2[indx] + "         CPU: " + output3[indx] +
				"         ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * ReLUDerivative section
		 * ---------------------
		 */

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.5f;
			output2[i] = 0.5f;
			output3[i] = 0.5f;
			activations[i] = -0.5f;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);
		acID1 = GPU.prepareFloatArray(deviceID1, activations, 0);
		acID2 = GPU.prepareFloatArray(deviceID2, activations, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);
		actID1 = GPU.prepareBuf(acID1, dimensions);
		actID2 = GPU.prepareBuf(acID2, dimensions);

		/* prepare kernels */
		// int TanhDID1 = GPU.TanhDerivative(deviceID1, ioaID1, actID1, 10);
		// int TanhDID2 = GPU.TanhDerivative(deviceID2, ioaID2, actID2, 10);

		int TanhDID1 = GPU.TanhDerivative(deviceID1, ioaID1, actID1, 10,
				TanhDerivative_activationsStartIndex, TanhDerivative_resultStartIndex);
		int TanhDID2 = GPU.TanhDerivative(deviceID2, ioaID2, actID2, 10,
				TanhDerivative_activationsStartIndex, TanhDerivative_resultStartIndex);

		job = TanhDID1 + "  " + TanhDID2 + // async start 2 kernels
				" !" + TanhDID1 + " !" + TanhDID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));

		for (int i = TanhDerivative_resultStartIndex; i < output3.length; i++)
		{
			float activation = activations[TanhDerivative_activationsStartIndex + id];
			output3[i] = output3[i] * (1 - activation * activation);
		}

		System.out.print("TanhDerivative      result[" + indx + "] device1: " + output1[indx] +
				"        device2: " + output2[indx] + "       CPU: " + output3[indx] +
				"       ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * SigmoidDerivative section
		 * ---------------------
		 */

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.5f;
			output2[i] = 0.5f;
			output3[i] = 0.5f;
			activations[i] = -0.5f;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);
		acID1 = GPU.prepareFloatArray(deviceID1, activations, 0);
		acID2 = GPU.prepareFloatArray(deviceID2, activations, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);
		actID1 = GPU.prepareBuf(acID1, dimensions);
		actID2 = GPU.prepareBuf(acID2, dimensions);

		/* prepare kernels */
		// int SigDID1 = GPU.SigmoidDerivative(deviceID1, ioaID1, actID1, 10);
		// int SigDID2 = GPU.SigmoidDerivative(deviceID2, ioaID2, actID2, 10);

		int SigDID1 = GPU.SigmoidDerivative(deviceID1, ioaID1, actID1, 10,
				SigmoidDerivative_activationsStartIndex, SigmoidDerivative_resultStartIndex);
		int SigDID2 = GPU.SigmoidDerivative(deviceID2, ioaID2, actID2, 10,
				SigmoidDerivative_activationsStartIndex, SigmoidDerivative_resultStartIndex);

		job = SigDID1 + "  " + SigDID2 + // async start 2 kernels
				" !" + SigDID1 + " !" + SigDID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));

		for (int i = SigmoidDerivative_resultStartIndex; i < output3.length; i++)
		{
			float activation = activations[SigmoidDerivative_activationsStartIndex + id];
			output3[i] = output3[i] * activation * (1 - activation);
		}

		System.out.print("SigmoidDerivative   result[" + indx + "] device1: " + output1[indx] +
				"       device2: " + output2[indx] + "      CPU: " + output3[indx] +
				"      ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * SoftReLUDerivative section
		 * ---------------------
		 */

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.5f;
			output2[i] = 0.5f;
			output3[i] = 0.5f;
			activations[i] = -0.5f;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);
		acID1 = GPU.prepareFloatArray(deviceID1, activations, 0);
		acID2 = GPU.prepareFloatArray(deviceID2, activations, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);
		actID1 = GPU.prepareBuf(acID1, dimensions);
		actID2 = GPU.prepareBuf(acID2, dimensions);

		/* prepare kernels */
		// int SReLUDID1 = GPU.SoftReLUDerivative(deviceID1, ioaID1, actID1, 10);
		// int SReLUDID2 = GPU.SoftReLUDerivative(deviceID2, ioaID2, actID2, 10);

		int SReLUDID1 = GPU.SoftReLUDerivative(deviceID1, ioaID1, actID1, 10,
				SoftReLUDerivative_activationsStartIndex, SoftReLUDerivative_resultStartIndex);
		int SReLUDID2 = GPU.SoftReLUDerivative(deviceID2, ioaID2, actID2, 10,
				SoftReLUDerivative_activationsStartIndex, SoftReLUDerivative_resultStartIndex);

		job = SReLUDID1 + "  " + SReLUDID2 + // async start 2 kernels
				" !" + SReLUDID1 + " !" + SReLUDID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));

		for (int i = SoftReLUDerivative_resultStartIndex; i < output3.length; i++)
		{
			output3[i] = output3[i] * (1 / (1 +
					(float) Math.exp(-activations[i + SoftReLUDerivative_activationsStartIndex])));
		}

		System.out.print("SoftReLUDerivative  result[" + indx + "] device1: " + output1[indx] +
				"   device2: " + output2[indx] + "  CPU: " + output3[indx] +
				"  ");
		if ((output1[indx] == output3[indx]) && (output2[indx] == output3[indx]))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * prng section
		 * ---------------------
		 */

		checkError(GPU.initPRNG(deviceID1, 10, 7)); // prepare 10 PRNGs, series # 7
		checkError(GPU.initPRNG(deviceID2, 10, 7)); // prepare 10 PRNGs, series # 7


		/*
		 * ---------------------
		 * prng subsection
		 * ---------------------
		 */

		checkError(GPU.prngRestart(deviceID1)); // just restart the prng (actually do not required)
		checkError(GPU.prngRestart(deviceID2)); // just restart the prng (actually do not required)

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0;
			output2[i] = 0;
			output3[i] = 0;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);

		/* prepare kernels */
		int prngID1 = GPU.prng(deviceID1, ioaID1, 10);
		int prngID2 = GPU.prng(deviceID2, ioaID2, 10);

		job = prngID1 + "  " + prngID2 + // async start 2 kernels
				" !" + prngID1 + " !" + prngID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));


		zr = 0.6806975f;
		System.out.print("prng                result[" + indx + "] device1: " + output1[indx] +
				"    device2: " + output2[indx] + "   CPU: " + zr +
				"   ");
		if ((output1[indx] == zr) && (output2[indx] == zr))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * prngGaussian section
		 * ---------------------
		 */

		/* prng initialization is performed in previous section */
		// checkError( GPU.initPRNG( deviceID1, 10, 7 ) ); // prepare 10 PRNGs, series # 7
		// checkError( GPU.initPRNG( deviceID2, 10, 7 ) ); // prepare 10 PRNGs, series # 7

		checkError(GPU.prngRestart(deviceID1)); // just restart the prng (actually do not required)
		checkError(GPU.prngRestart(deviceID2)); // just restart the prng (actually do not required)

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0;
			output2[i] = 0;
			output3[i] = 0;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);

		/* prepare kernels */
		int prngGaussianID1 = GPU.prngGaussian(deviceID1, ioaID1, 10);
		int prngGaussianID2 = GPU.prngGaussian(deviceID2, ioaID2, 10);

		job = prngGaussianID1 + "  " + prngGaussianID2 + // async start 2 kernels
				" !" + prngGaussianID1 + " !" + prngGaussianID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));


		{ // calculate next prn with Gaussian distribution
			float v1 = 0.361395f; // 1st prn (0.053564638f)
			float v2 = 0.28899777f; // 2nd prn (0.93228686f)
			float s = v1 * v1 + v2 * v2;
			zr = v1 * (float) Math.sqrt(-2 * (float) Math.log(s) / s);
		}
		System.out.print("prngGaussian        result[" + indx + "] device1: " + output1[indx] +
				"    device2: " + output2[indx] + "   CPU: " + zr +
				"   ");
		if ((output1[indx] == zr) && (output2[indx] == zr))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * Noise section
		 * ---------------------
		 */

		/* prng initialization is performed in previous section */
		// checkError( GPU.initPRNG( deviceID1, 10, 7 ) ); // prepare 10 PRNGs, series # 7
		// checkError( GPU.initPRNG( deviceID2, 10, 7 ) ); // prepare 10 PRNGs, series # 7

		checkError(GPU.prngRestart(deviceID1)); // just restart the prng (actually do not required)
		checkError(GPU.prngRestart(deviceID2)); // just restart the prng (actually do not required)

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.1f;
			output2[i] = 0.1f;
			output3[i] = 0.5f;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.cloneFloatBuf(ioID1, deviceID2); // clone buffer from deviceID1 into deviceID2
//		ioID2 = GPU.prepareFloatArray( deviceID2, output2, 0 );

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);

		/* prepare kernels */
		// int NoiseID1 = GPU.Noise(deviceID1, ioaID1, 10);
		// int NoiseID2 = GPU.Noise(deviceID2, ioaID2, 10);

		int NoiseID1 = GPU.Noise(deviceID1, ioaID1, 10, Noise_startIndex,
				Noise_corruptionLevel, Noise_corruptedValue);
		int NoiseID2 = GPU.Noise(deviceID2, ioaID2, 10, Noise_startIndex,
				Noise_corruptionLevel, Noise_corruptedValue);

		/* update values in the local arrays */
//		for ( int i = 0; i < output1.length; i++ )
//		{
//			output1[i] = 0.5f;
//			output2[i] = 0.5f;
//		}

		/* update buffers */
//		checkError( GPU.updateFloatBuf( ioaID1, output1 ) );
//		checkError( GPU.updateFloatBuf( ioaID2, output2 ) );

		job = NoiseID1 + "  " + NoiseID2 + // async start 2 kernels
				" !" + NoiseID1 + " !" + NoiseID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
// checkError( GPU.copyFloatBuf( ioaID1, ioaID2 ) ); // copy results from ioaID1 into ioaID2
		checkError(GPU.getFloatBuf(ioaID2, output2));

//		for( int i = 0; i < output1.length; i++ )
//		{
//			System.out.print( output1[i] + " " );
//		}
//		System.out.println("");


		zr = 0.1f; // here prng = 0.735259
		System.out.print("Noise               result[" + indx + "] device1: " + output1[indx] +
				"          device2: " + output2[indx] + "         CPU: " + zr +
				"         ");
		if ((output1[indx] == zr) && (output2[indx] == zr))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * BernoulliKernel section
		 * ---------------------
		 */

		/* prng initialization is performed in previous section */
		// checkError( GPU.initPRNG( deviceID1, 10, 7 ) ); // prepare 10 PRNGs, series # 7
		// checkError( GPU.initPRNG( deviceID2, 10, 7 ) ); // prepare 10 PRNGs, series # 7

		checkError(GPU.prngRestart(deviceID1)); // just restart the prng (actually do not required)
		checkError(GPU.prngRestart(deviceID2)); // just restart the prng (actually do not required)

		for (int i = 0; i < output1.length; i++)
		{
			output1[i] = 0.5f;
			output2[i] = 0.5f;
			output3[i] = 0.5f;
		}

		/* prepare buffers */
		ioID1 = GPU.prepareFloatArray(deviceID1, output1, 0);
		ioID2 = GPU.prepareFloatArray(deviceID2, output2, 0);

		/* prepare arrays */
		ioaID1 = GPU.prepareBuf(ioID1, dimensions);
		ioaID2 = GPU.prepareBuf(ioID2, dimensions);

		/* prepare kernels */
		int BKID1 = GPU.BernoulliKernel(deviceID1, ioaID1, 10);
		int BKID2 = GPU.BernoulliKernel(deviceID2, ioaID2, 10);

		job = BKID1 + "  " + BKID2 + // async start 2 kernels
				" !" + BKID1 + " !" + BKID2; // wait for kernels finish
		checkError(GPU.kernelRunJob(job.toCharArray())); // push jobs to devices

		/* get results */
		checkError(GPU.getFloatBuf(ioaID1, output1));
		checkError(GPU.getFloatBuf(ioaID2, output2));


		zr = (output3[indx] > 0.735259) ? 1 : 0; // here prng = 0.735259
		System.out.print("BernoulliKernel     result[" + indx + "] device1: " + output1[indx] +
				"          device2: " + output2[indx] + "         CPU: " + zr +
				"         ");
		if ((output1[indx] == zr) && (output2[indx] == zr))
			System.out.println("[CORRECT!]");
		else
			System.out.println("[ERROR!]");


		/*
		 * ---------------------
		 * conv2DFF section
		 * ---------------------
		 */

		for (int i = 0; i < 32; i++)
		{
			if (i < 19)
			{
				input[i] = (i + 1);
			}
			else
			{
				input[i] = 0;
			}
		}

//		weights[0]=1;	weights[1]=2;	weights[2]=3;	weights[3]=4;
//		weights[4]=1;	weights[5]=2;	weights[6]=3;	weights[7]=4;


//		int ciID1 = GPU.prepareFloatArray( deviceID1, input,   0 );
//		int cwID1 = GPU.prepareFloatArray( deviceID1, weights, 0 );
//		int coID1 = GPU.prepareFloatArray( deviceID1, output1, 0 );

//		int ciID2 = GPU.prepareFloatArray( deviceID2, input,   0 );
//		int cwID2 = GPU.prepareFloatArray( deviceID2, weights, 0 );
//		int coID2 = GPU.prepareFloatArray( deviceID2, output2, 0 );


//		int[] featureMapOffsets = new int[10];
//		int fID1 = GPU.prepareIntConstArray( deviceID1, featureMapOffsets, 0 );


		/*
		 * ---------------------
		 * Finalization section
		 * ---------------------
		 */

		checkError(GPU.finalizeDeviceAll());

		System.out.println("");
		System.out.println("_[DONE]_");
	}

	public static void checkError(int result)
	{
		if (result != 0)
		{
			System.out.println("[ERROR!] " + result);
		}
	}

	public static int getSize(int[] dimensions)
	{
		int inputSize = 1;
		for (int i = 0; i < dimensions.length; i++)
		{
			if (dimensions[i] > 0)
			{
				inputSize *= dimensions[i];
			}
		}

		return inputSize;
	}
}