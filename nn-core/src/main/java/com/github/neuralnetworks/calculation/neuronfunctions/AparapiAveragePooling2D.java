package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.memory.ValuesProvider;

/**
 * Average pooling
 */

public class AparapiAveragePooling2D implements ConnectionCalculator {

    private static final long serialVersionUID = 8165829315701496713L;

    private AparapiAveragePooling2DCC cc;

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (cc == null || cc.getMiniBatchSize() != valuesProvider.getMiniBatchSize()) {
	    cc = new AparapiAveragePooling2DCC((Subsampling2DConnection) connections.get(0), valuesProvider, targetLayer);
	}

	cc.calculate(connections, valuesProvider, targetLayer);
    }

    public static class AparapiAveragePooling2DCC extends AparapiSubsampling2D {

	private static final long serialVersionUID = -2393526660090301257L;

	public AparapiAveragePooling2DCC(Subsampling2DConnection c, ValuesProvider valuesProvider, Layer targetLayer) {
	    super(c, valuesProvider, targetLayer);
	}

	@Override
	protected void pool(int inputStartIndex, int outputStartIndex) {
	    float sum = 0;

	    for (int i = 0; i < miniBatchSize; i++) {
		sum = 0;
		for (int j = 0; j < regionLength; j++) {
		    sum += input[inputStartIndex + featureMapOffsets[i * regionLength + j]];
		}

		output[outputStartIndex + i * outputMiniBatchDistance] = sum / regionLength;
	    }
	}
    }
}