package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.List;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.calculation.ValuesProvider;

/**
 * Average pooling
 */

public class AparapiAveragePooling2D implements ConnectionCalculator {

    private static final long serialVersionUID = 8165829315701496713L;

    private ConnectionCalculator cc;

    @Override
    public void calculate(List<Connections> connections, ValuesProvider valuesProvider, Layer targetLayer) {
	if (cc == null) {
	    cc = new AparapiAveragePooling2DCC((Subsampling2DConnection) connections.get(0), valuesProvider.getColumns());
	}

	cc.calculate(connections, valuesProvider, targetLayer);
    }

    public static class AparapiAveragePooling2DCC extends AparapiSubsampling2D {

	private static final long serialVersionUID = -2393526660090301257L;

	public AparapiAveragePooling2DCC(Subsampling2DConnection c, int miniBatchSize) {
	    super(c, miniBatchSize);
	}

	@Override
	protected void pool(int inputStartIndex) {
	    int rl = regionLength;
	    int miniBatch = miniBatchSize;
	    float sum = 0;

	    for (int i = 0; i < miniBatch; i++) {
		sum = 0;
		for (int j = 0; j < rl; j++) {
		    sum += input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i];
		}

		output[getGlobalId() * miniBatch + i] = sum / rl;
	    }
	}
    }
}