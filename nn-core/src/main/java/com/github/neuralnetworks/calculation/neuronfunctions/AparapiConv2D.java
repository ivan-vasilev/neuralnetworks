package com.github.neuralnetworks.calculation.neuronfunctions;

import java.io.Serializable;
import java.util.Arrays;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.Util;

/**
 * Base class for convolutional operations (2d)
 * This connection accept as input a single training example (as opposed to the weighted sum which works with multiple).
 *
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 */
public abstract class AparapiConv2D extends Kernel implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * input
     */
    protected float[] input;
    protected final int inputStartIndex;
    protected final int inputFeatureMapRowsDistance;
    protected final int inputFeatureMapColumnsDistance;

    /**
     * output
     */
    protected float[] output;
    protected final int outputStartIndex;
    protected final int outputFeatureMapsDistance;
    protected final int outputFeatureMapRowsDistance;
    protected final int outputFeatureMapColumnsDistance;
    protected final int outputMiniBatchDistance;
    protected final int outputFeatureMapLength; // output columns * output rows
    protected final int outputColumns; // output column count (columns of image for example)

    /**
     * combined feature weights of all feature maps
     */
    //@Local TODO
    protected final float[] weights;
    protected final int weightsStartIndex;

    /**
     * weights for single feature map
     */
    protected final int featureMapWeights;

    /**
     * input offset for each feature map in respect to the start index
     */
    //@Local TODO
    @Constant
    protected final int[] featureMapOffsets;

    /**
     * number of samples per calculation (for example number of images)
     */
    protected final int miniBatchSize;

    /**
     * stride
     */
    protected final int stride;

    public AparapiConv2D(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
	super();

	Tensor input = null, output = null;
	if (targetLayer == c.getOutputLayer()) {
	    input = valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c);
	    output = valuesProvider.getValues(targetLayer, c);
	} else {
	    input = valuesProvider.getValues(targetLayer, c);
	    output = valuesProvider.getValues(Util.getOppositeLayer(c, targetLayer), c);
	}

	this.input = input.getElements();
	this.inputStartIndex = input.getStartIndex();
	this.inputFeatureMapRowsDistance = input.getDimensionElementsDistance(1);
	this.inputFeatureMapColumnsDistance = input.getDimensionElementsDistance(2);

	this.output = output.getElements();
	this.outputStartIndex =  output.getStartIndex();
	this.outputFeatureMapsDistance =  output.getDimensionElementsDistance(0);
	this.outputFeatureMapRowsDistance = output.getDimensionElementsDistance(1);
	this.outputFeatureMapColumnsDistance = output.getDimensionElementsDistance(2);
	this.outputMiniBatchDistance = output.getDimensionElementsDistance(3);

	this.weights = c.getWeights().getElements();
	this.weightsStartIndex = c.getWeights().getStartIndex();

	this.miniBatchSize = valuesProvider.getMiniBatchSize();
	this.outputColumns = c.getOutputFeatureMapColumns();
	this.outputFeatureMapLength = c.getOutputFeatureMapLength();
	this.stride = c.getStride();
	this.featureMapWeights = c.getKernelColumns() * c.getKernelRows() * c.getInputFilters();
	this.featureMapOffsets = new int[featureMapWeights * miniBatchSize];

	int inputMiniBatchDistance = input.getDimensionElementsDistance(3);
	int inputFeatureMapsDistance =  input.getDimensionElementsDistance(0);

	for (int m = 0, offset = 0; m < miniBatchSize; m++) {
	    for (int i = 0; i < c.getInputFilters(); i++) {
		for (int j = 0; j < c.getKernelRows(); j++) {
		    for (int k = 0; k < c.getKernelColumns(); k++) {
			featureMapOffsets[offset++] = i * inputFeatureMapsDistance + j * inputFeatureMapRowsDistance + k * inputFeatureMapColumnsDistance + m * inputMiniBatchDistance;
		    }
		}
	    }
	}
    }

    public void calculate(Conv2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
	if (c != null) {
	    Environment.getInstance().getExecutionStrategy().execute(this, targetLayer.getUnitCount(Arrays.asList(new Conv2DConnection[] {c})));
	}
    }

    @Override
    public void run() {
	int id = getGlobalId();

	conv(	weightsStartIndex + featureMapWeights * (id / outputFeatureMapLength),
		inputStartIndex + ((id % outputFeatureMapLength) / outputColumns) * inputFeatureMapRowsDistance * stride + (id % outputColumns) * inputFeatureMapColumnsDistance * stride,
		outputStartIndex + (id / outputFeatureMapLength) * outputFeatureMapsDistance + ((id % outputFeatureMapLength) / outputColumns) * outputFeatureMapRowsDistance + (id % outputColumns) * outputFeatureMapColumnsDistance);
    }

    /**
     * the actual convolution
     * @param weightsStartId
     * @param inputStartId
     * @param outputStartId
     */
    protected void conv(int weightsStartId, int inputStartId, int outputStartId) {
    }
}
