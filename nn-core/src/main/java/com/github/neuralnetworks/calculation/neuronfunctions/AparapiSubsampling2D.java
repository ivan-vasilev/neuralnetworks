package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.Util;

/**
 * Base Aparapi connection calculator for subsampling layers.
 * 
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 */
public abstract class AparapiSubsampling2D extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = 8931101094464503687L;

    /**
     * input samples count
     */
    protected final int miniBatchSize;

    /**
     * subsampling region rows
     */
    protected final int subsamplingRows;
    
    /**
     * subsampling region columns
     */
    protected final int subsamplingCols;

    /**
     * Length of the subsampling region (subsampling rows *  subsampling cols)
     */
    protected final int regionLength;

    /**
     * offset from start when mapping input to output
     */
    protected final int ioRowsOffset;

    /**
     * offset from start when mapping input to output
     */
    protected final int ioColumnsOffset;

    /**
     * Contains the offset in the input array for each cell of the current region. The offset is calculated in respect to the first cell of the region
     */
    //@Local TODO
    protected final int[] featureMapOffsets;

    /**
     * input data
     */
    protected float[] input;
    protected final int inputStartIndex;
    protected final int inputFeatureMapsDistance;
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
    protected final int outputFeatureMapLength;
    protected final int outputFeatureMapColumns;
    protected final int outputMiniBatchDistance;

    public AparapiSubsampling2D(Subsampling2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
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
	this.inputFeatureMapsDistance =  input.getDimensionElementsDistance(0);
	this.inputFeatureMapRowsDistance = input.getDimensionElementsDistance(1);
	this.inputFeatureMapColumnsDistance = input.getDimensionElementsDistance(2);

	this.output = output.getElements();
	this.outputStartIndex =  output.getStartIndex();
	this.outputFeatureMapsDistance =  output.getDimensionElementsDistance(0);
	this.outputFeatureMapRowsDistance = output.getDimensionElementsDistance(1);
	this.outputFeatureMapColumnsDistance = output.getDimensionElementsDistance(2);
	this.outputMiniBatchDistance = output.getDimensionElementsDistance(3);
	this.outputFeatureMapLength = c.getOutputFeatureMapLength();
	this.outputFeatureMapColumns = c.getOutputFeatureMapColumns();

	this.miniBatchSize = input.getDimensionLength(3);

	this.subsamplingRows = c.getSubsamplingRegionRows();
	this.subsamplingCols = c.getSubsamplingRegionCols();
	this.regionLength = c.getSubsamplingRegionLength();
	this.ioRowsOffset = (c.getInputFeatureMapRows() % subsamplingRows) / 2;
	this.ioColumnsOffset = (c.getInputFeatureMapColumns() % subsamplingCols) / 2;

	this.featureMapOffsets = new int[regionLength * miniBatchSize];
	int inputMiniBatchDistance = input.getDimensionElementsDistance(3);
	for (int m = 0, i = 0; m < miniBatchSize; m++) {
	    for (int j = 0; j < subsamplingRows; j++) {
		for (int k = 0; k < subsamplingCols; k++) {
		    featureMapOffsets[i++] = j * inputFeatureMapRowsDistance + k * inputFeatureMapColumnsDistance + m * inputMiniBatchDistance;
		}
	    }
	}
    }

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (connections.size() > 0) {
	    Subsampling2DConnection c = (Subsampling2DConnection) connections.get(0);
	    if (targetLayer == c.getOutputLayer()) {
		Environment.getInstance().getExecutionStrategy().execute(this, targetLayer.getUnitCount(connections));
	    } else {
		Environment.getInstance().getExecutionStrategy().execute(this, Util.getOppositeLayer(c, targetLayer).getUnitCount(connections));
	    }
	}
    }

    /* (non-Javadoc)
     * @see com.amd.aparapi.Kernel#run()
     * input start index is calculated and passed to the pooling method
     */
    @Override
    public void run() {
	int id = getGlobalId();

	// get current offset
	int fm = id / outputFeatureMapLength;
	int fmOffset = id % outputFeatureMapLength;
	int fmRow = fmOffset / outputFeatureMapColumns;
	int fmCol = fmOffset % outputFeatureMapColumns;

	pool(	inputStartIndex + fm * inputFeatureMapsDistance + (ioRowsOffset + fmRow * subsamplingRows) * inputFeatureMapRowsDistance + (ioColumnsOffset + fmCol * subsamplingCols) * inputFeatureMapColumnsDistance,
		outputStartIndex + fm * outputFeatureMapsDistance + fmRow * outputFeatureMapRowsDistance + fmCol * outputFeatureMapColumnsDistance);
    }

    /**
     * This is where the subsampling happens
     */
    protected void pool(int inputStartIndex, int outputStartIndex) {
    }

    public int getMiniBatchSize() {
        return miniBatchSize;
    }
}
