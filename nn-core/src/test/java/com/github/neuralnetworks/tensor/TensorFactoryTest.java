package com.github.neuralnetworks.tensor;

import org.junit.Test;

import com.github.neuralnetworks.tensor.Tensor;
import com.github.neuralnetworks.tensor.TensorFactory;
import com.github.neuralnetworks.test.AbstractTest;

import static org.junit.Assert.*;
/**
 * Created by chass on 18.11.14.
 */
public class TensorFactoryTest extends AbstractTest {

    @Test
    public void testTensorConstructor01(){

        Tensor tensor = TensorFactory.tensor(1, 2, 3);
        assertEquals(3, tensor.getDimensions().length);
        assertEquals(1,tensor.getDimensions()[0]);
        assertEquals(2,tensor.getDimensions()[1]);
        assertEquals(3,tensor.getDimensions()[2]);

    }

    // tensor(float[] elements, int offset, int... dimensions)

    @Test
    public void testTensorConstructor02(){

        float[] elements = new float[]{-2,-1,0,1,2,3,4,5,6,7,8,9};
        Tensor tensor = TensorFactory.tensor(elements, 2, 3,3);
        assertEquals(2, tensor.getDimensions().length);
        assertEquals(3,tensor.getDimensions()[0]);
        assertEquals(3,tensor.getDimensions()[1]);
        assertEquals(2,tensor.getStartIndex());

        int count = 0;
        for(int x=0;x<3;x++){
            for(int y=0;y<3;y++){
                assertEquals(count,tensor.get(x,y),0);
                count++;
            }
        }

    }


//    tensor(Tensor parent, int[][] dimensionsLimit, boolean reduceChildDimensions)

    @Test
    public void testTensorConstructor03(){

        float[] elements = new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

        Tensor parent = TensorFactory.tensor(elements, 0, 5,5);


        Tensor submatrix = TensorFactory.tensor(parent,new int[][]{{1,1},{3,3}},false);
        assertEquals(2, submatrix.getDimensions().length);
        assertEquals(3,submatrix.getDimensions()[0]);
        assertEquals(3,submatrix.getDimensions()[1]);


        Tensor subvector = TensorFactory.tensor(parent,new int[][]{{0,0},{0,4}},true);
        assertEquals(1, subvector.getDimensions().length);
        assertEquals(5,subvector.getDimensions()[0]);

        assertEquals(0,subvector.get(0),0);
        assertEquals(1,subvector.get(1),0);
        assertEquals(2,subvector.get(2),0);
        assertEquals(3,subvector.get(3),0);
        assertEquals(4,subvector.get(4),0);

    }

    // Tensor[] tensor(int[]... dimensions)

    // Matrix matrix(float[][] elements)

    //  Matrix matrix(float[] elements, int columns)

    // static void fill(Tensor t, float value)

    // ValuesProvider tensorProvider(NeuralNetwork nn, int miniBatchSize, boolean useSharedMemory)

    // 	public static ValuesProvider tensorProvider(int miniBatchSize, boolean useSharedMemory, NeuralNetwork... nns)



}
