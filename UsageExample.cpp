
MemImage* image = ...;
Octree octree(image, 12);
if (octree.isUsable())
{
	if (octree.setInsideRange(minIntesity, maxIntensity))
		octree.enumerate();
	
	const vector<int>& cubes = octree.getCubesInside();	
	performOptimizedRendering(image, cubes);
}
else
{
	performUnoptimizedRendering(image);
}
