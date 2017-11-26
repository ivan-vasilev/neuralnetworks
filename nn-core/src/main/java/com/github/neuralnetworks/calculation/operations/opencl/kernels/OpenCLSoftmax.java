package com.github.neuralnetworks.calculation.operations.opencl.kernels;

import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLArrayReferenceManager;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLCore;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLKernelReference;
import com.github.neuralnetworks.calculation.operations.opencl.OpenCLTensorFunction;
import com.github.neuralnetworks.tensor.Matrix;
import com.github.neuralnetworks.util.Environment;

public class OpenCLSoftmax extends OpenCLTensorFunction
{
	private static final long serialVersionUID = 1L;

	@Override
	public OpenCLKernelReference createKernel()
	{
		this.deviceId = this.deviceId != null ? this.deviceId : Environment.getInstance().getRuntimeConfiguration().getOpenCLConfiguration().getPreferredDevice();

		OpenCLArrayReferenceManager rm = OpenCLArrayReferenceManager.getInstance();

		OpenCLArrayReference tensorRef = rm.getArrayReference(tensor, deviceId);
		if (tensorRef == null) {
			rm.addToDevice(deviceId, tensor);
			tensorRef = rm.getArrayReference(tensor, deviceId);
		}

		Matrix m = (Matrix) tensor;
		int id = OpenCLCore.getInstance().SoftmaxFunction(deviceId, tensorRef.getId(), m.getRows(), m.getStartIndex(), m.getColumns(), m.getRowElementsDistance(), m.getColumnElementsDistance());
		return new OpenCLKernelReference(deviceId, id, getModifiedArrays());
	}

	@Override
	public String kernelOptions(int order)
	{
		Matrix m = (Matrix) tensor;
		return " -D SfSI" + order + "=" + m.getStartIndex() + " -D Sfc" + order + "=" + m.getColumns() + " -D SfnRS" + order + "=" + m.getRowElementsDistance() + " -D SfnCS" + order + "=" + m.getColumnElementsDistance() + " ";
	}
}
