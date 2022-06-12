#include <Fusion/Base/Octree.h>
#include <Fusion/Base/Timer.h>
#include <Fusion/Base/Log.h>


namespace Fusion
{
	namespace {
		class ThreadAbortedException : public std::exception {
		public:
			ThreadAbortedException() = default;
		};
	}

	Octree::Octree(int minCubeSize) :
		m_minCubeSize(minCubeSize),
		m_min(std::numeric_limits<int>::min()),
		m_max(std::numeric_limits<int>::max()),
		m_scale(1.0),
		m_voxelsInside(0),
		m_thread(0),
		m_usable(false)
	{
	}

	Octree::Octree(MemImage* image, int minCubeSize) :
		m_minCubeSize(minCubeSize),
		m_min(std::numeric_limits<int>::min()),
		m_max(std::numeric_limits<int>::max()),
		m_scale(1.0),
		m_voxelsInside(0),
		m_thread(0),
		m_abortThread(false),
		m_usable(false)
	{
		m_thread = new std::thread(&Octree::setImage, this, image);
	}

	void Octree::setImage(MemImage* image)
	{
		m_usable = false;

		Timer t;
		m_scale = (double)((1 << (8 * image->typeSize())) - 1);
		int nx = createLayerGrid(image->width(), m_gridX);
		int ny = createLayerGrid(image->height(), m_gridY);
		int nz = createLayerGrid(image->slices(), m_gridZ);
		m_numLayers = std::max(std::max(nx, ny), nz);
		LOG_DEBUG("Octree layers " << nx << " x " << ny << " x " << nz);

		// Fill up smaller dimensions, if applicable
		while (nx < m_numLayers) { m_gridX.push_back(m_gridX[nx - 1]); nx++; }
		while (ny < m_numLayers) { m_gridY.push_back(m_gridY[ny - 1]); ny++; }
		while (nz < m_numLayers) { m_gridZ.push_back(m_gridZ[nz - 1]); nz++; }

		// Allocate data
		int lx = 0, ly = 0, lz = 0;
		for (int i = 0; i < m_numLayers; i++) {
			int size = (int)(m_gridX[lx].size() * m_gridY[ly].size() * m_gridZ[lz].size());
			m_data.push_back(new OctreeElement[size]);
			if (lx < nx - 1) lx++;
			if (ly < ny - 1) ly++;
			if (lz < nz - 1) lz++;
		}

		// Fill the damn thing
		try
		{
			if (image->type() == Image::USHORT)
				fill<unsigned short>(reinterpret_cast<TypedImage<unsigned short>*>(image));
			else if (image->type() == Image::UBYTE)
				fill<unsigned char>(reinterpret_cast<TypedImage<unsigned char>*>(image));

			LOG_DEBUG("Octree computation completed in " << t.passed() << " ms");

			m_usable = true;
		}
		catch (ThreadAbortedException&)
		{
			LOG_DEBUG("Octree computation aborted");
		}
	}

	Octree::~Octree()
	{
		if (m_thread)
		{
			m_abortThread = true;
			m_thread->join();
			delete m_thread;
		}
	}


	int Octree::createLayerGrid(int dim, std::vector<std::vector<int> >& grid) {
		int cubeSizeHalf = dim / 2;
		int layer = 0;
		std::vector<int> firstLayer;
		firstLayer.push_back(dim);
		grid.push_back(firstLayer);
		while (cubeSizeHalf >= m_minCubeSize)
		{
			std::vector<int> newLayer;
			for (int i = 0; i < (int)grid[layer].size(); i++)
			{
				int size = grid[layer][i];
				int halfSize = size / 2;
				newLayer.push_back(halfSize);
				newLayer.push_back(size - halfSize);
			}
			grid.push_back(newLayer);
			cubeSizeHalf /= 2;
			layer++;
		}
		return layer + 1;
	}


	template<typename T> void Octree::fill(TypedImage<T>* image) {
		m_usable = false;

		std::vector<int> lx = m_gridX[m_numLayers - 1];
		std::vector<int> ly = m_gridY[m_numLayers - 1];
		std::vector<int> lz = m_gridZ[m_numLayers - 1];
		int nx = (int)lx.size();
		int ny = (int)ly.size();
		int nz = (int)lz.size();
		T* imgPtr = image->pointer();
		int elementPos = 0;

		// Fill the highest layer from image data
		int pz = 0;
		for (int z = 0; z < nz; z++)
		{
			int py = 0;

			if (m_abortThread)
				throw ThreadAbortedException();

			for (int y = 0; y < ny; y++) {
				int px = 0;
				for (int x = 0; x < nx; x++) {
					OctreeElement& element = m_data[m_numLayers - 1][elementPos++];
					for (int zz = 0; zz < lz[z]; zz++) {
						for (int yy = 0; yy < ly[y]; yy++) {
							for (int xx = 0; xx < lx[x]; xx++) {
								int value = (int)imgPtr[px + xx + image->width() * (py + yy + image->height() * (pz + zz))];
								if (element.min > value) element.min = value;
								if (element.max < value) element.max = value;
							}
						}
					}
					px += lx[x];
				}
				py += ly[y];
			}
			pz += lz[z];
		}

		// Propagate up to the other layers
		for (int layer = m_numLayers - 2; layer >= 0; layer--)
		{
			nx = (int)m_gridX[layer].size();
			ny = (int)m_gridY[layer].size();
			nz = (int)m_gridZ[layer].size();
			elementPos = 0;
			for (int z = 0; z < nz; z++) {
				if (m_abortThread)
					throw ThreadAbortedException();

				for (int y = 0; y < ny; y++) {
					for (int x = 0; x < nx; x++) {
						// Determine if one or two cubes per dimension are present in the layer below
						int nxx, nyy, nzz; getSplit(layer, nxx, nyy, nzz);
						// Fill element properties from its children
						OctreeElement& elementUp = m_data[layer][elementPos++];
						for (int zz = 0; zz < nzz; zz++) {
							for (int yy = 0; yy < nyy; yy++) {
								for (int xx = 0; xx < nxx; xx++) {
									int indexDown = nxx * x + xx + nx * nxx * (nyy * y + yy + ny * nyy * (nzz * z + zz));
									OctreeElement& elementDown = m_data[layer + 1][indexDown];
									if (elementUp.min > elementDown.min) elementUp.min = elementDown.min;
									if (elementUp.max < elementDown.max) elementUp.max = elementDown.max;
								}
							}
						}
					}
				}
			}
		}

		m_usable = true;
	}


	bool Octree::setInsideRange(int min, int max) {
		if ((m_min == min) && (m_max == max))
			return false;
		m_min = min; m_max = max;
		m_voxelsInside = 0;
		Timer t;
		// Recurse into Octree
		checkChildren(0, 0, 0, 0);
		// Print statistics
		int numVoxels = m_gridX[0][0] * m_gridY[0][0] * m_gridZ[0][0];
		double percentage = 100.0 * (double)m_voxelsInside / (double)numVoxels;
		LOG_DEBUG("Octree range [" << m_min << ".." << m_max << "], " << percentage << "% inside, " << t.passed() << " ms");
		return true;
	}


	bool Octree::setInsideRange(double min, double max) {
		// TODO: More appropriate rounding
		return setInsideRange(int(min * m_scale), int(max * m_scale));
	}


	Octree::ElementType Octree::checkChildren(int layer, int px, int py, int pz) {
		int nx = (int)m_gridX[layer].size();
		int ny = (int)m_gridY[layer].size();
		OctreeElement& element = m_data[layer][px + nx * (py + ny * pz)];
		if ((m_min > element.max) || (m_max < element.min))
			// Current element is outside of requested range, return at any level
			element.type = LEAF_OUT;
		else if (layer == m_numLayers - 1) {
			// Element is inside and on the last level
			element.type = LEAF_IN;
			// Update statistics
			m_voxelsInside += m_gridX[layer][px] * m_gridY[layer][py] * m_gridZ[layer][pz];
		}
		else {
			// Something else, need to check children
			int nxx, nyy, nzz; getSplit(layer, nxx, nyy, nzz);
			bool allIn = true, allOut = true;
			for (int zz = 0; zz < nzz; zz++) {
				for (int yy = 0; yy < nyy; yy++) {
					for (int xx = 0; xx < nxx; xx++) {
						ElementType value = checkChildren(layer + 1, nxx * px + xx, nyy * py + yy, nzz * pz + zz);
						if (value == LEAF_IN)		allOut = false;
						else if (value == LEAF_OUT) allIn = false;
						else { allIn = false; allOut = false; }
					}
				}
			}
			if (allIn)		 element.type = LEAF_IN;
			else if (allOut) element.type = LEAF_OUT; // This should never happen, but won't hurt to check
			else			 element.type = NODE;
		}
		return element.type;
	}


	const std::vector<int>& Octree::enumerate() {
		m_cubesInside.clear();
		Timer t;
		int num = enumerateChildren(0, 0, 0, 0);
		LOG_DEBUG("Octree has " << num << " cubes");
		return m_cubesInside;
	}


	int Octree::enumerateChildren(int layer, int px, int py, int pz) {
		int count = 0;
		int nx = (int)m_gridX[layer].size();
		int ny = (int)m_gridY[layer].size();
		OctreeElement& element = m_data[layer][px + nx * (py + ny * pz)];
		if (element.type == LEAF_IN) {
			// Compute voxel position of this cube, TODO: use pre-computed arrays
			int vx = 0, vy = 0, vz = 0; int i;
			for (i = 0; i < px; i++)
				vx += m_gridX[layer][i];
			for (i = 0; i < py; i++)
				vy += m_gridY[layer][i];
			for (i = 0; i < pz; i++)
				vz += m_gridZ[layer][i];
			// Add this cube
			m_cubesInside.push_back(vx);
			m_cubesInside.push_back(vy);
			m_cubesInside.push_back(vz);
			m_cubesInside.push_back(m_gridX[layer][px]);
			m_cubesInside.push_back(m_gridY[layer][py]);
			m_cubesInside.push_back(m_gridZ[layer][pz]);
			count++;
		}
		else if (element.type == NODE) {
			// Node, need to check children
			int nxx, nyy, nzz; getSplit(layer, nxx, nyy, nzz);
			for (int zz = 0; zz < nzz; zz++)
				for (int yy = 0; yy < nyy; yy++)
					for (int xx = 0; xx < nxx; xx++)
						count += enumerateChildren(layer + 1, nxx * px + xx, nyy * py + yy, nzz * pz + zz);
		}
		return count;
	}

}
