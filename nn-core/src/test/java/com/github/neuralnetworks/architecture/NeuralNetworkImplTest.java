package com.github.neuralnetworks.architecture;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.test.AbstractTest;

import static org.junit.Assert.assertEquals;

/**
 * Created by chass on 02.12.14.
 */
public class NeuralNetworkImplTest extends AbstractTest {

    @Test
    public void testRemoveLayer()
    {
        NeuralNetworkImpl mlp = NNFactory.mlp(new int[]{3, 4, 5}, true);
        assertEquals(5, mlp.getLayers().size(), 0);
        Layer currentOutput = mlp.getOutputLayer();
        mlp.removeLayer(mlp.getOutputLayer());
        assertEquals(3, mlp.getLayers().size(), 0);
        assertEquals(true, currentOutput != mlp.getOutputLayer());
    }
}
