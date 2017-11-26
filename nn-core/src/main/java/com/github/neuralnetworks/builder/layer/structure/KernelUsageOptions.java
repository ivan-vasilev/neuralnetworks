package com.github.neuralnetworks.builder.layer.structure;

/**
 * @author tmey
 */
public interface KernelUsageOptions
{

	public void setPaddingSize(int size);

	public void setStrideSize(int size);

	public void setStrideRows(int strideRows);

	public void setStrideColumns(int strideColumns);

	public void setPaddingRows(int paddingRows);

	public void setPaddingColumns(int paddingColumns);
}
