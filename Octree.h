#ifndef FUSION_OCTREE_H
#define FUSION_OCTREE_H

// **Note to the reader who does not have access to TypedImage.h:**
// Both, MemImage and TypedImage<T> are classes representing 2D/3D images. 
// MemImage describes the abstract interface.
// TypedImage<T> inherits from MemImage and implements it for a concrete element type T.
#include <Fusion/Base/TypedImage.h>

#include <limits>
#include <thread>
#include <vector>
#include <atomic>

namespace Fusion
{
	/// Fast Octree space subdivision
	class Octree {
	public:
		/// Creates the Octree with the specified smallest cube size
		Octree(int minCubeSize);

		/// Creates and fills the Octree in a background thread with given image and minimum cube size
		Octree(MemImage* image, int minCubeSize); // I would not accept raw pointers. Use shared_ptr instead in this place and other similar places.

		/// Destructor, deletes the Octree
		~Octree();

		/// Set the intensity range defining 'inside' and update
		/** Returns true if something has changed. */
		bool setInsideRange(int min, int max);

		/// Convenience method, set range with normalized scale (0..1)
		bool setInsideRange(double min, double max);

		/// Enumerate all inside cube cells with their position and size
		const std::vector<int>& enumerate();

		const std::vector<int>& getCubesInside() const { return m_cubesInside; }

		/// Fast template method to (re-)fill Octree from image data
		template<typename T> void fill(TypedImage<T>* image);

		/// Fill Octree from image data
		void setImage(MemImage* image);

		/// Tells if the Octree is computed and ready to use
		bool isUsable() const { return m_usable; }

	protected:
		enum ElementType {
			NODE,		///< Children with mixed conditions
			LEAF_IN,	///< All children satisfy the condition
			LEAF_OUT	///< All children violate the condition
		};

		struct OctreeElement {
			OctreeElement() :
				min(std::numeric_limits<int>::max()),
				max(std::numeric_limits<int>::min()),
				type(NODE) {}

			int min;
			int max;
			ElementType type;
		};

		/// Creates element layer subdivision given the size of an individual image dimension
		int createLayerGrid(int dim, std::vector<std::vector<int> >& grid);

		/// Recursively check and update Octree children for range condition
		ElementType checkChildren(int layer, int px, int py, int pz);

		/// Recursively enumerate Octree children which are inside
		int enumerateChildren(int layer, int px, int py, int pz);

		/// For better readability, get the split between current and next layer
		inline void getSplit(int layer, int& sx, int& sy, int& sz) const {
			sx = (int)m_gridX[layer + 1].size() / (int)m_gridX[layer].size(); // I would cast using static_cast in this place and any other similar places
			sy = (int)m_gridY[layer + 1].size() / (int)m_gridY[layer].size();
			sz = (int)m_gridZ[layer + 1].size() / (int)m_gridZ[layer].size();
		}

		int m_minCubeSize;						///< The smallest allowed octree cell dimension
		int m_numLayers;						///< The number of layers of the octree
		std::vector<std::vector<int> > m_gridX;	///< Cell size in x for every layer
		std::vector<std::vector<int> > m_gridY;	///< Cell size in y for every layer
		std::vector<std::vector<int> > m_gridZ;	///< Cell size in z for every layer
		std::vector<OctreeElement *>   m_data;	///< Cell data for every layer
		int m_min;								///< Desired minimum value for range testing
		int m_max;								///< Desired maximum value for range testing
		double m_scale;							///< Scale for conversion to integer intensities
		std::vector<int> m_cubesInside;			///< List of all cube coordinates classified as inside
		int m_voxelsInside;						///< Number of voxels satisfying range condition
		std::thread* m_thread; // I would make it a unique_ptr					///< Thread for background creation of octree
		std::atomic<bool> m_abortThread;						///< Flag whether to abort the computation 
		bool m_usable;							///< The octree is filled and ready to use if true
	};

}

#endif
