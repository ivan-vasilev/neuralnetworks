package com.github.neuralnetworks.util;



/**
 * Simple matrix representation with one-dimensional array. This is required,
 * because Aparapi supports only one-dim arrays (otherwise the execution is
 * transferred to the cpu)
 */
public class Matrix extends Tensor {

    private static final long serialVersionUID = 1L;

    private int[] dimTmp;

    public Matrix() {
	super();
    }

    public Matrix(float[] elements, int columns) {
	super(elements, elements.length / columns, columns);
	dimTmp = new int[dimensions.length];
    }

    public Matrix(int rows, int columns) {
	super(rows, columns);
	dimTmp = new int[dimensions.length];
    }

    public Matrix(Matrix copy) {
	super(copy.getRows(), copy.getColumns());
	dimTmp = new int[dimensions.length];
    }

    public Matrix(Tensor parent, int[] dimStart, int[] dimEnd) {
	super(parent, dimStart, dimEnd);
	dimTmp = new int[dimensions.length];
    }

    public int getColumns() {
	return getDimension(getColumnsDimension());
    }

    protected int getColumnsDimension() {
	int d = getRowsDimension() + 1;
	for (int i = getDimensions().length - 1; i > 0; i--) {
	    if (dimStart[i] != dimEnd[i] && d < i) {
		d = i;
		break;
	    }
	}

	return d;
    }

    public int getRows() {
	return getDimension(getRowsDimension());
    }

    protected int getRowsDimension() {
	int d = getDimensions().length - 2;
	for (int i = 0; i < getDimensions().length - 1; i++) {
	    if (dimStart[i] != dimEnd[i]) {
		d = i;
		break;
	    }
	}

	return d;
    }

    /* (non-Javadoc)
     * d[0] - rows
     * d[1] - columns
     * @see com.github.neuralnetworks.util.Tensor#get(int[])
     */
    @Override
    public float get(int... d) {
	if (d.length != 2) {
	    throw new IllegalArgumentException("Please provide row and column only");
	}

	dimTmp[getRowsDimension()] = d[0];
	dimTmp[getColumnsDimension()] = d[1];

	return super.get(dimTmp);
    }

    @Override
    public void set(float value, int... d) {
	if (d.length != 2) {
	    throw new IllegalArgumentException("Please provide row and column only");
	}

	dimTmp[getRowsDimension()] = d[0];
	dimTmp[getColumnsDimension()] = d[1];

	super.set(value, dimTmp);
    }
}
