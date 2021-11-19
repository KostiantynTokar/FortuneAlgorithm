#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>
#include <numbers>
#include <cassert>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;
using namespace py::literals;

bool isClose(double a, double b, double maxRelDiff = 1e-9, double maxAbsDiff = 1e-10)
{
	const auto diff = abs(a - b);
	return diff <= maxRelDiff * abs(a)
		|| diff <= maxRelDiff * abs(b)
		|| diff <= maxAbsDiff;
}

bool isCloseToZero(double a, double maxAbsDiff = 1e-10)
{
	return isClose(a, 0.0, 0.0, maxAbsDiff);
}

bool definitelyGreaterThan(double a, double b, double epsilon = numeric_limits<double>::epsilon())
{
	return (a - b) > (max(abs(a), abs(b)) * epsilon);
}

bool definitelyLessThan(double a, double b, double epsilon = numeric_limits<double>::epsilon())
{
	return (b - a) > (max(abs(a), abs(b)) * epsilon);
}

struct Point
{
	double x;
	double y;
};

struct DoublyConnectedEdgeList
{
	struct Vertex
	{
		Point p;
		size_t halfEdge; // Index of one of the half-edges that have this vertex as an endpoint.
	};
	struct Face
	{
		size_t halfEdge; // Index of one of the half-edges that surround this face.
	};
	struct HalfEdge
	{
		ptrdiff_t vertexFrom = -1; // -1 for half-edges that starts from infinite.
		
		size_t face; // Index of a face to the left of this edge.

		ptrdiff_t next = -1; // -1 if there is no next (infinite half-edge).

		// NOTE: twin half-edges located in sequence in halfEdges array, i.e., halfEdges[2 * i] and halfEdges[2 * i + 1] are twins.
		// NOTE: prev = twin->next->twin->next->twin
	};
	vector<HalfEdge> halfEdges; // For n >= 3 sites, up to 3 * n - 6 edges, so up to 6 * n - 12 half-edges.
	vector<Vertex> vertices; // For n >= 3 sites, up to 2 * n - 5 vertices.
	vector<Face> faces;
};

void addEdge(DoublyConnectedEdgeList& dcel, const size_t s1, const size_t s2)
{
	dcel.faces[s1] = { .halfEdge = dcel.halfEdges.size() };
	dcel.faces[s2] = { .halfEdge = dcel.halfEdges.size() + 1 };
	dcel.halfEdges.push_back({ .face = s1 });
	dcel.halfEdges.push_back({ .face = s2 });
}

// Returns x-coordinate of an intersection of the arc1 from the left and arc2 from the right.
double parabolasIntersectionX(const double sweepLineY, const Point& site1, const Point& site2)
{
	const auto sx1 = site1.x;
	const auto sx2 = site2.x;
	const auto sy1 = site1.y;
	const auto sy2 = site2.y;
	if (isClose(sy1, sy2))
	{
		return (sx1 + sx2) / 2;
	}
	const auto p1 = site1.y - sweepLineY;
	const auto p2 = site2.y - sweepLineY;

	const auto mb = sx1 * p2 - sx2 * p1;

	const auto pDiff = p2 - p1;
	const auto changeSign = pDiff < 0;
	const auto sxDiff = sx2 - sx1;
	const auto Dsqr = p1 * p2 * (sxDiff * sxDiff + pDiff * pDiff);
	const auto D = Dsqr > 0.0 ? sqrt(Dsqr) : 0.0;
	return (changeSign ? sy1 < sy2 : sy1 > sy2)
		? (mb - D) / pDiff
		: (mb + D) / pDiff;
}

struct BeachLine
{
	// Implemented as AVL tree.

	struct Node
	{
		Node* parent;
		Node* left;
		Node* right;
		size_t p; // For intersections: site that defines left arc. For leaves: site that defines the arc.
		size_t q; // For intersections: site that defines right arc. 
		ptrdiff_t halfEdge = -1; // For intersections: index of a half-edge (one of two twins) that is built from this intersection.
		ptrdiff_t circleEventId; // -1 if no circle event that is connected to this leaf.
		signed char balance; // Difference in height between right subtree and left subtree.

		constexpr bool isLeaf() const
		{
			return left == nullptr && right == nullptr;
		}

		constexpr bool isLeft() const
		{
			assert(parent != nullptr);
			return parent->left == this;
		}
		constexpr bool isRight() const
		{
			assert(parent != nullptr);
			return parent->right == this;
		}

		constexpr bool isRootOrLeft() const
		{
			return parent == nullptr ? true : isLeft();
		}
		constexpr bool isRootOrRight() const
		{
			return parent == nullptr ? true : isRight();
		}
	};

	Node* root;
	const std::vector<Point>& sites;

	constexpr BeachLine(const std::vector<Point>& sites, std::vector<size_t> sitesWithTheBiggestY, DoublyConnectedEdgeList& dcel)
		: sites{ sites }
		, root{ nullptr }
	{
		assert(sitesWithTheBiggestY.size() >= 1);
		std::sort(
			std::begin(sitesWithTheBiggestY), std::end(sitesWithTheBiggestY),
			[&sites](const size_t lhs, const size_t rhs)
			{
				return sites[lhs].x < sites[rhs].x;
			});
		root = init(std::cbegin(sitesWithTheBiggestY), std::cend(sitesWithTheBiggestY), dcel);
	}

	~BeachLine()
	{
		deleteRecursive(root);
	}

	template<typename It>
	constexpr Node* init(It begin, It end, DoublyConnectedEdgeList& dcel)
	{
		assert(begin != end);
		const auto dist = std::distance(begin, end);
		if (dist == 1)
		{
			return new Node{ .parent = nullptr, .left = nullptr, .right = nullptr, .p = *begin, .q = 0, .circleEventId = -1, .balance = 0 };
		}
		const auto mid = dist / 2;
		const auto s1 = *std::next(begin, mid - 1);
		const auto s2 = *std::next(begin, mid);
		const auto n = new Node{
			.parent = nullptr,
			.left = init(begin, std::next(begin, mid), dcel),
			.right = init(std::next(begin, mid), end, dcel),
			.p = s1, .q = s2,
			.halfEdge = static_cast<ptrdiff_t>(dcel.halfEdges.size()),
			.circleEventId = -1,
			.balance = static_cast<signed char>(mid * 2 == dist ? 0 : 1)
		};
		n->left->parent = n;
		n->right->parent = n;
		addEdge(dcel, s1, s2);
		return n;
	}

	constexpr Node* findRegion(const size_t site) const
	{
		return findRegionFrom(root, site, sites[site].y);
	}

	constexpr Node* findRegionFrom(Node* node, const size_t site, const double sweepLinePos) const
	{
		assert(node != nullptr);
		if (node->isLeaf())
			return node;

		const auto intersectionX = parabolasIntersectionX(sweepLinePos, sites[node->p], sites[node->q]);

		// If intersection's x-coordinate equals to site's x-coordinate, then new circle event will be created and instantly processed
		// (piece of left arc with is created and removed instantly).
		// Then the same will happen to the right arc of this intersection.
		const auto nextNode = sites[site].x <= intersectionX ? node->left : node->right;
		return findRegionFrom(nextNode, site, sweepLinePos);
	}

	struct NewArcInfo
	{
		ptrdiff_t id; // id of an arc that has beed intersected.
		Node* left;
		Node* leftCentralIntersection;
		Node* central;
		Node* centralRightIntersection;
		Node* right;
	};

	constexpr NewArcInfo insertArc(const size_t site)
	{
		assert(!empty());
		auto regionNode = findRegion(site);
		const auto intersectedArc = regionNode->p;
		const auto eventId = regionNode->circleEventId;
		auto regionNodeParent = regionNode->parent;

		auto newSubTree = new Node{ .parent = regionNodeParent, .left = nullptr, .right = nullptr, .p = intersectedArc, .q = site, .circleEventId = -1, .balance = 0 };
		newSubTree->left = regionNode;
		newSubTree->left->parent = newSubTree;
		newSubTree->left->circleEventId = -1;
		assert(newSubTree->left->halfEdge == -1);
		newSubTree->right = new Node{ .parent = newSubTree, .left = nullptr, .right = nullptr, .p = site, .q = intersectedArc, .circleEventId = -1, .balance = 0 };
		const auto redLeaf = newSubTree->right; // Rebalance, then add to this node 2 children.

		auto res = NewArcInfo{
			.id = eventId,
			.left = newSubTree->left,
			.leftCentralIntersection = newSubTree,
			.centralRightIntersection = newSubTree->right,
		};
		
		replaceNode(regionNode, newSubTree, regionNodeParent);

		auto child = newSubTree;
		if (regionNodeParent != nullptr)
		{
			do
			{
				if (regionNodeParent->left == child)
					--regionNodeParent->balance;
				else
					++regionNodeParent->balance;
				assert(-2 <= regionNodeParent->balance && regionNodeParent->balance <= 2);
				child = regionNodeParent;
				regionNodeParent = regionNodeParent->parent;
				rebalanceSubTree(child, regionNodeParent);
			} while (regionNodeParent != nullptr && child->balance != 0);
		}

		redLeaf->left = new Node{ .parent = redLeaf, .left = nullptr, .right = nullptr, .p = site, .q = 0, .circleEventId = -1, .balance = 0 };
		redLeaf->right = new Node{ .parent = redLeaf, .left = nullptr, .right = nullptr, .p = intersectedArc, .q = 0, .circleEventId = -1, .balance = 0 };

		res.central = redLeaf->left;
		res.right = redLeaf->right;

		child = redLeaf;
		regionNodeParent = child->parent;
		if (regionNodeParent != nullptr)
		{
			do
			{
				if (regionNodeParent->left == child)
					--regionNodeParent->balance;
				else
					++regionNodeParent->balance;
				assert(-2 <= regionNodeParent->balance && regionNodeParent->balance <= 2);
				child = regionNodeParent;
				regionNodeParent = regionNodeParent->parent;
				rebalanceSubTree(child, regionNodeParent);
			} while (regionNodeParent != nullptr && child->balance != 0);
		}

		return res;
	}

	// Returns node with new intersection (between leaf to the left of node and leaf to the right of node).
	constexpr Node* removeArc(Node* node)
	{
		assert(node->isLeaf());
		const auto [leftIntersection, leftHeight] = findIntersectionWithLeftLeaf(node);
		const auto [rightIntersection, rightHeight] = findIntersectionWithRightLeaf(node);
		assert(leftHeight == 1 || rightHeight == 1);

		const auto s1 = leftIntersection->p;
		const auto s2 = rightIntersection->q;

		const auto [higherNode, lowerNode] = leftHeight > rightHeight ? make_tuple(leftIntersection, rightIntersection) : make_tuple(rightIntersection, leftIntersection);

		higherNode->p = s1;
		higherNode->q = s2;

		const auto insteadLower = lowerNode->left == node ? lowerNode->right : lowerNode->left;
		auto aboveLower = lowerNode->parent;
		if (aboveLower->left == lowerNode)
		{
			aboveLower->left = insteadLower;
			++aboveLower->balance;
		}
		else
		{
			aboveLower->right = insteadLower;
			--aboveLower->balance;
		}
		insteadLower->parent = lowerNode->parent;
		delete lowerNode;
		delete node;

		auto parent = aboveLower->parent;
		rebalanceSubTree(aboveLower, parent);

		while (parent != nullptr && aboveLower->balance == 0)
		{
			// Height of subtree rooted at aboveLower is decreased.
			if (parent->left == aboveLower)
				++parent->balance;
			else
				--parent->balance;
			aboveLower = parent;
			parent = parent->parent;
			rebalanceSubTree(aboveLower, parent);
		}

		return higherNode;
	}

	constexpr static tuple<Node*, int> findIntersectionWithLeftLeaf(Node* node)
	{
		assert(node->isLeaf());
		int height = 1;
		while (node->isRootOrLeft())
		{
			node = node->parent;
			++height;
			// This check is not necessary in insertArc.
			if (node == nullptr)
				return make_tuple(nullptr, 0);
		}
		return make_tuple(node->parent, height);
	}

	constexpr static tuple<Node*, int> findIntersectionWithRightLeaf(Node* node)
	{
		assert(node->isLeaf());
		int height = 1;
		while (node->isRootOrRight())
		{
			node = node->parent;
			++height;
			// This check is not necessary in insertArc.
			if (node == nullptr)
				return make_tuple(nullptr, 0);
		}
		return make_tuple(node->parent, height);
	}

	constexpr static Node* findLeafToLeftFromIntersection(Node* node)
	{
		assert(!node->isLeaf());
		if (node == nullptr)
			return nullptr;
		node = node->left;
		while (!node->isLeaf())
			node = node->right;
		return node;
	}

	constexpr static Node* findLeafToRightFromIntersection(Node* node)
	{
		assert(!node->isLeaf());
		if (node == nullptr)
			return nullptr;
		node = node->right;
		while (!node->isLeaf())
			node = node->left;
		return node;
	}

	/*constexpr static Node* findLeafToLeft(Node* node)
	{
		assert(node->isLeaf());
		const auto [intersection, _] = findIntersectionWithLeftLeaf(node);
		return findLeafToLeftFromIntersection(intersection);
	}

	constexpr static Node* findLeafToRight(Node* node)
	{
		assert(node->isLeaf());
		const auto [intersection, _] = findIntersectionWithRightLeaf(node);
		return findLeafToRightFromIntersection(intersection);
	}*/

	constexpr bool empty() const
	{
		return root == nullptr;
	}

	constexpr void rebalanceSubTree(Node*& node, Node* parent)
	{
		if (node->balance == 2)
		{
			auto child = node->right;
			if (child->balance >= 0)
			{
				// Left left rotation.
				replaceNode(node, child, parent);
				node->parent = child;
				node->right = child->left;
				child->left->parent = node;
				child->left = node;

				node->balance = 1 - child->balance;
				--child->balance;

				node = child;
			}
			else
			{
				// Left right rotation.
				auto grandchild = child->left;
				replaceNode(node, grandchild, parent);
				node->parent = grandchild;
				node->right = grandchild->left;
				grandchild->left->parent = node;
				child->parent = grandchild;
				child->left = grandchild->right;
				grandchild->right->parent = child;
				grandchild->left = node;
				grandchild->right = child;

				if (grandchild->balance == 1)
					node->balance = -1;
				else
					node->balance = 0;
				if (grandchild->balance == -1)
					child->balance = 1;
				else
					child->balance = 0;
				grandchild->balance = 0;

				node = grandchild;
			}
		}
		else if (node->balance == -2)
		{
			auto child = node->left;
			if (child->balance <= 0)
			{
				// Right right rotation.
				replaceNode(node, child, parent);
				node->parent = child;
				node->left = child->right;
				child->right->parent = node;
				child->right = node;

				node->balance = -1 - child->balance;
				++child->balance;

				node = child;
			}
			else
			{
				// Right left rotation.
				auto grandchild = child->right;
				replaceNode(node, grandchild, parent);
				node->parent = grandchild;
				node->left = grandchild->right;
				grandchild->right->parent = node;
				child->parent = grandchild;
				child->right = grandchild->left;
				grandchild->left->parent = child;
				grandchild->right = node;
				grandchild->left = child;

				if (grandchild->balance == 1)
					child->balance = -1;
				else
					child->balance = 0;
				if (grandchild->balance == -1)
					node->balance = 1;
				else
					node->balance = 0;
				grandchild->balance = 0;

				node = grandchild;
			}
		}
	}

	constexpr void trustedReplaceNode(Node* oldNode, Node* newNode, Node* parent)
	{
		assert(parent);
		if (parent->left == oldNode)
			parent->left = newNode;
		else
			parent->right = newNode;
		newNode->parent = parent;
	}

	constexpr void replaceNode(Node* oldNode, Node* newNode, Node* parent)
	{
		if (parent)
		{
			trustedReplaceNode(oldNode, newNode, parent);
		}
		else
		{
			root = newNode;
			root->parent = nullptr;
		}
	}

	constexpr void deleteRecursive(Node* node)
	{
		if (node == nullptr) return;
		deleteRecursive(node->left);
		deleteRecursive(node->right);
		delete node;
	}
};

struct Event
{
	enum class Type
	{
		site,
		circle
	};
	double y;
	Point center; // For circle events;
	BeachLine::Node* leaf; // For circle events.
	Type type;
};

struct PriorityQueue
{
	// Implemented as binary heap
	 
	// Or it can be an array of structs that contain Event and its id.
	Event* storage;
	size_t* ids;
	size_t size;
	size_t newId;
	vector<size_t> idToInd;

	constexpr PriorityQueue(const vector<Point>& sites)
		: storage{ new Event[sites.size() * 2] }
		, ids{ new size_t[sites.size() * 2] }
		, size{ sites.size() }
		, newId{ sites.size() }
	{
		idToInd.reserve(sites.size());
		for (size_t i{ 0 }; i != sites.size(); ++i)
		{
			storage[i] = Event{ .y = sites[i].y, .type = Event::Type::site };
			ids[i] = i;
			idToInd.push_back(i);
		}
		for (size_t i{ sites.size() / 2 + 1 }; i != 0; --i)
		{
			downHeapify(i - 1);
		}
	}

	~PriorityQueue()
	{
		delete[] ids;
		delete[] storage;
	}

	constexpr size_t insertCircleEvent(const double y, const Point& center, BeachLine::Node* leaf)
	{
		assert(leaf->isLeaf());
		ids[size] = newId;
		idToInd.push_back(size);
		storage[size] = Event{ .y = y, .center = center, .leaf = leaf, .type = Event::Type::circle };
		++size;
		upHeapify(size - 1);
		return newId++;
	}

	constexpr void removeById(const size_t id)
	{
		assert(!empty());
		const auto ind = idToInd[id];
		auto oldKey = storage[ind].y;
		auto newKey = storage[size - 1].y;
		moveFromLast(ind);
		if (newKey > oldKey)
		{
			upHeapify(ind);
		}
		else
		{
			downHeapify(ind);
		}
	}

	// Returns first event in the queue and its id. Id of a site event is its index in the vertices vector.
	constexpr tuple<Event, size_t> pop()
	{
		assert(!empty());
		const auto res = make_tuple(storage[0], ids[0]);
		moveFromLast(0);
		downHeapify(0);
		return res;
	}

	constexpr bool empty() const
	{
		return size == 0;
	}

	constexpr void upHeapify(size_t c)
	{
		auto p = (c - 1) / 2;
		while (c != 0 && storage[p].y < storage[c].y)
		{
			swap(p, c);
			c = p;
			p = (c - 1) / 2;
		}
	}

	constexpr void downHeapify(size_t p)
	{
		while (true)
		{
			auto left = 2 * p + 1;
			auto right = 2 * p + 2;
			auto largest = p;
			if (left < size && storage[left].y > storage[largest].y)
				largest = left;
			if (right < size && storage[right].y > storage[largest].y)
				largest = right;
			if (largest == p)
				return;
			swap(p, largest);
			p = largest;
		}
	}

	constexpr void moveFromLast(const size_t to)
	{
		assert(!empty());
		--size;
		storage[to] = storage[size];
#ifdef _DEBUG
		idToInd[ids[to]] = numeric_limits<size_t>::max();
#endif
		idToInd[ids[size]] = to;
		ids[to] = ids[size];
	}

	constexpr void swap(const size_t i, const size_t j)
	{
		assert(i < size);
		assert(j < size);
		auto tmp = storage[i];
		storage[i] = storage[j];
		storage[j] = tmp;
		idToInd[ids[i]] = j;
		idToInd[ids[j]] = i;
		auto tmpId = ids[i];
		ids[i] = ids[j];
		ids[j] = tmpId;
	}
};

// Returns y-coordinate of the bottom point and the center of the circle.
tuple<double, Point> circleBottomPoint(const Point& a, const Point& b, const Point& c)
{
	// Intersection of perpendiculars that pass through the center of a circle of lines
	// (b.x - a.x) * y - (b.y - a.y) * x + a.y * (b.x - a.x) - a.x = 0
	// (c.x - b.x) * y - (c.y - b.y) * x + b.y * (c.x - b.x) - b.x = 0
	const auto y21 = b.y - a.y;
	const auto y32 = c.y - b.y;
	const auto y31 = c.y - a.y;
	const auto y12s = a.y + b.y;
	const auto y23s = b.y + c.y;
	const auto x21 = b.x - a.x;
	const auto x32 = c.x - b.x;
	const auto x31 = c.x - a.x;
	const auto x12s = a.x + b.x;
	const auto x23s = b.x + c.x;
	const auto denom = 2 * (y32 * x21 - y21 * x32);
	const auto xc = -(y21 * y32 * y31 - y32 * x21 * x12s + y21 * x32 * x23s) / denom;
	const auto yc =  (x21 * x32 * x31 - x32 * y21 * y12s + x21 * y32 * y23s) / denom;
	const auto r = hypot(a.x - xc, a.y - yc);
	return make_tuple(yc - r, Point{ xc, yc });
}

bool isConvergent(const double y, const Point& siteLeft, const Point& siteCentral, const Point& siteRight)
{
	// True if triangle if left-oriented.
	return definitelyLessThan((siteCentral.x - siteLeft.x) * (siteRight.y - siteLeft.y) - (siteCentral.y - siteLeft.y) * (siteRight.x - siteLeft.x), 0);
}

DoublyConnectedEdgeList fortune(const vector<Point>& sites)
{
	auto dcel = DoublyConnectedEdgeList{};

	dcel.halfEdges.reserve(sites.size() >= 3 ? 6 * sites.size() - 12 : (sites.size() == 2 ? 2 : 0));
	dcel.vertices.reserve(sites.size() >= 3 ? 2 * sites.size() - 5 : 0);
	dcel.faces.resize(sites.size());

	auto queue = PriorityQueue{ sites };

	if (queue.empty())
	{
		return dcel;
	}

	auto sitesWithTheBiggestY = vector<size_t>{};
	do
	{
		sitesWithTheBiggestY.push_back(get<1>(queue.pop()));
	} while (!queue.empty() && isClose(queue.storage[0].y, sites[sitesWithTheBiggestY.back()].y));
	
	auto beachLine = BeachLine{ sites, std::move(sitesWithTheBiggestY), dcel };

	const auto createCircleEvents = [&sites, &queue, &beachLine]
	(const double y,
		BeachLine::Node* left, BeachLine::Node* leftIntersection, BeachLine::Node* innerLeft,
		BeachLine::Node* innerRight, BeachLine::Node* rightIntersection, BeachLine::Node* right)
	{
		if(const auto leftleftIntersection = get<0>(beachLine.findIntersectionWithLeftLeaf(left)))
		{
			if(isConvergent(y, sites[leftleftIntersection->p], sites[leftleftIntersection->q], sites[innerLeft->p]))
			{
				const auto leftleftSite = leftleftIntersection->p;
				assert(leftleftSite != left->p && leftleftSite != innerLeft->p && left->p != innerLeft->p);
				const auto [bottom, center] = circleBottomPoint(sites[leftleftSite], sites[left->p], sites[innerLeft->p]);
				left->circleEventId = queue.insertCircleEvent(bottom, center, left);
			}
		}
		if (const auto rightrightIntersection = get<0>(beachLine.findIntersectionWithRightLeaf(right)))
		{
			if(isConvergent(y, sites[innerRight->p], sites[rightrightIntersection->p], sites[rightrightIntersection->q]))
			{
				const auto rightrightSite = rightrightIntersection->q;
				assert(rightrightSite != right->p && rightrightSite != innerRight->p && right->p != innerRight->p);
				const auto [bottom, center] = circleBottomPoint(sites[rightrightSite], sites[right->p], sites[innerRight->p]);
				right->circleEventId = queue.insertCircleEvent(bottom, center, right);
			}
		}
	};

	while (!queue.empty())
	{
		const auto [ev, evId] = queue.pop();
		switch (ev.type)
		{
		break; case Event::Type::site:
		{
			const auto [intersectedArcEventId, left, leftCentral, central, centralRight, right] = beachLine.insertArc(evId);
			if (intersectedArcEventId != -1)
			{
				queue.removeById(intersectedArcEventId);
			}
			assert(central->p != left->p && central->p != right->p);
			assert(left->p == right->p);
			assert(left->p == leftCentral->p && central->p == leftCentral->q);
			assert(central->p == centralRight->p && right->p == centralRight->q);
			centralRight->halfEdge = leftCentral->halfEdge = dcel.halfEdges.size();
			addEdge(dcel, central->p, left->p);
			createCircleEvents(ev.y, left, leftCentral, central, central, centralRight, right);
		}
		break; case Event::Type::circle:
		{
			const auto arcToRemove = ev.leaf;
			const auto leftIntersection = get<0>(beachLine.findIntersectionWithLeftLeaf(arcToRemove));
			const auto rightIntersection = get<0>(beachLine.findIntersectionWithRightLeaf(arcToRemove));
			const auto left = beachLine.findLeafToLeftFromIntersection(leftIntersection);
			const auto right = beachLine.findLeafToRightFromIntersection(rightIntersection);
			assert(left != nullptr);
			assert(right != nullptr);
			assert(arcToRemove->p != left->p);
			assert(arcToRemove->p != right->p);
			if (left->circleEventId != -1)
			{
				queue.removeById(left->circleEventId);
				left->circleEventId = -1;
			}
			if (right->circleEventId != -1)
			{
				queue.removeById(right->circleEventId);
				right->circleEventId = -1;
			}

			assert(leftIntersection->halfEdge != -1);
			assert(rightIntersection->halfEdge != -1);

			const auto sCentral = arcToRemove->p;
			const auto leftIntersectionHalfEdge = leftIntersection->halfEdge;
			const auto rightIntersectionHalfEdge = rightIntersection->halfEdge;
			const auto intersection = beachLine.removeArc(arcToRemove);

			const auto sLeft = left->p;
			const auto sRight = right->p;

			const auto intersectionHalfEdge = intersection->halfEdge = dcel.halfEdges.size();

			dcel.halfEdges.push_back({ .face = sLeft });
			dcel.halfEdges.push_back({ .face = sRight });

			const auto setVertexFrom = [&dcel](const ptrdiff_t edgeInd, const size_t s)
			{
				if (dcel.halfEdges[edgeInd].face == s)
				{
					dcel.halfEdges[edgeInd].vertexFrom = dcel.vertices.size();
				}
				else
				{
					dcel.halfEdges[edgeInd + 1].vertexFrom = dcel.vertices.size();
				}
			};
			
			setVertexFrom(leftIntersectionHalfEdge, sLeft);
			setVertexFrom(rightIntersectionHalfEdge, sCentral);
			assert(dcel.halfEdges[intersectionHalfEdge + 1].face == sRight);
			dcel.halfEdges[intersectionHalfEdge + 1].vertexFrom = dcel.vertices.size();

			dcel.vertices.push_back({
				.p = ev.center,
				.halfEdge = static_cast<size_t>(intersectionHalfEdge + 1)
			});

			const auto setNext = [&halfEdges = dcel.halfEdges]
			(const ptrdiff_t centralHalfEdge, const ptrdiff_t leftHalfEdge, const size_t sLeft)
			{
				if (halfEdges[centralHalfEdge].face == sLeft)
				{
					halfEdges[centralHalfEdge].next = halfEdges[leftHalfEdge].face == sLeft ? leftHalfEdge : leftHalfEdge + 1;
				}
				else
				{
					halfEdges[centralHalfEdge + 1].next = halfEdges[leftHalfEdge].face == sLeft ? leftHalfEdge : leftHalfEdge + 1;
				}
			};

			setNext(intersectionHalfEdge, leftIntersectionHalfEdge, sLeft);
			setNext(leftIntersectionHalfEdge, rightIntersectionHalfEdge, sCentral);
			setNext(rightIntersectionHalfEdge, intersectionHalfEdge, sRight);

			createCircleEvents(ev.y, left, intersection, right, left, intersection, right);
		}
		}
	}

	return dcel;
}

template<typename Seq, typename DimSeq>
auto to_pyarray(Seq&& seq, DimSeq&& dimSeq)
{
	using ValSeq = std::remove_reference_t<Seq>;
	auto seq_ptr = new ValSeq(std::forward<Seq>(seq));
	auto capsule = py::capsule(seq_ptr, [](void* p) { delete reinterpret_cast<ValSeq*>(p); });
	return py::array(std::forward<DimSeq>(dimSeq), seq_ptr->data(), capsule);
}
template<typename Seq>
auto to_pyarray(Seq&& seq)
{
	const auto dim = seq.size();
	return to_pyarray(std::forward<Seq>(seq), std::vector<size_t>{dim});
}

int main()
{
	//const auto sites = vector<Point>{ {0, 10}, {1, 9}, {5, 8}, {3, 4}, {4, 5}, {1, -1}, {5, -2}, {-5, -5}, {-10, -6}, {-9, 2}, {-11, 7}, {-3, 0}, {-2, 6}, {-11, 11}, {-6, 11} };
	//const auto sites = vector<Point>{ {1, 2}, {0, 1.5}, {0, 1}, {0, 0.5}, {0, 0}, {0, -0.5}, {0, -1} };
	//const auto sites = vector<Point>{ {4, 0}, {0, 8}, {8, 2}, {7, 9} };
	//const auto sites = vector<Point>{ {-1, 1}, {1, 1}, {3, 1}, {0, 4}, {2, 4}, {-1, 7}, {1, 7}, {3, 7} };
	//const auto sites = vector<Point>{ {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {1, 0}, {2, 0}, {3, 0}, {1, 4}, {2, 4}, {3, 4} };
	//const auto sites = vector<Point>{ {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4},
	// 									{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4},
	// 									{1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4},
	// 									{2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4},
	// 									{3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}
	//};
	//const auto sites = vector<Point>{ {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4},
	//									{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4},
	//									{1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4},
	//									{3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}
	//};
	auto sites = vector<Point>{ {0, 0} };
	const auto sitesNumber = 20;
	for (int i{ 0 }; i != sitesNumber; ++i)
	{
		sites.push_back({ cos(2.0 * i * std::numbers::pi / sitesNumber), sin(2.0 * i * std::numbers::pi / sitesNumber) });
		sites.push_back({ 4.0 * cos(2.0 * i * std::numbers::pi / sitesNumber), 3.0 * sin(2.0 * i * std::numbers::pi / sitesNumber) });
		sites.push_back({ 5.0 * cos(2.0 * i * std::numbers::pi / sitesNumber), 7.0 * sin(2.0 * i * std::numbers::pi / sitesNumber) });
	}
	const auto vor = fortune(sites);
	const auto minMaxXY = [](const tuple<double, double, double, double>& accum, const Point& p)
	{
		return make_tuple(min(get<0>(accum), p.x), max(get<1>(accum), p.x), min(get<2>(accum), p.y), max(get<3>(accum), p.y));
	};
	const auto sitesMinMax = sites.size() == 0
		? make_tuple(0.0, 0.0, 0.0, 0.0)
		: reduce(
			cbegin(sites), cend(sites),
			make_tuple(numeric_limits<double>::max(), numeric_limits<double>::min(), numeric_limits<double>::max(), numeric_limits<double>::min()),
			minMaxXY
		);
	constexpr auto extendDrawRegionForAllVertices = true;
	const auto [minX, maxX, minY, maxY] = !extendDrawRegionForAllVertices
		? sitesMinMax
		: transform_reduce(
			cbegin(vor.vertices), cend(vor.vertices),
			sitesMinMax,
			minMaxXY,
			[](const auto& vertex) { return vertex.p; }
		);
	constexpr auto regionSizeMultiplier = 1.61803398875;
	const auto regionCenterX = (minX + maxX) / 2;
	const auto regionCenterY = (minY + maxY) / 2;
	const auto regionHalfSizeExact = regionSizeMultiplier * max(maxX - minX, maxY - minY) / 2;
	const auto regionHalfSize = isCloseToZero(regionHalfSizeExact) ? 0.5 : regionHalfSizeExact;
	const auto drawMinX = regionCenterX - regionHalfSize;
	const auto drawMaxX = regionCenterX + regionHalfSize;
	const auto drawMinY = regionCenterY - regionHalfSize;
	const auto drawMaxY = regionCenterY + regionHalfSize;

	vector<double> vertexXs;
	vector<double> vertexYs;
	for(const auto& p : vor.vertices)
	{
		vertexXs.push_back(p.p.x);
		vertexYs.push_back(p.p.y);
	}

	// First half of the vectors represents first endpoints of a line segment, second half - second endpoints.
	vector<size_t> edges;
	vector<double> infEdgeXs;
	vector<double> infEdgeYs;
	vector<double> doubleInfEdgeXs;
	vector<double> doubleInfEdgeYs;

	size_t edgeCount = 0;
	size_t infEdgeCount = 0;
	size_t doubleInfEdgeCount = 0;
	for (size_t i{ 0 }; i < vor.halfEdges.size(); i += 2)
	{
		const auto& e1 = vor.halfEdges[i];
		const auto& e2 = vor.halfEdges[i + 1];
		if (e1.vertexFrom != -1 && e2.vertexFrom != -1)
			++edgeCount;
		else if (e1.vertexFrom != -1 || e2.vertexFrom != -1)
			++infEdgeCount;
		else
			++doubleInfEdgeCount;
	}

	edges.resize(edgeCount * 2);
	infEdgeXs.resize(infEdgeCount * 2);
	infEdgeYs.resize(infEdgeCount * 2);
	doubleInfEdgeXs.resize(doubleInfEdgeCount * 2);
	doubleInfEdgeYs.resize(doubleInfEdgeCount * 2);

	size_t edgeCounter = 0;
	size_t infEdgeCounter = 0;
	size_t doubleInfEdgeCounter = 0;
	for (size_t i{ 0 }; i < vor.halfEdges.size(); i += 2)
	{
		const auto& e1 = vor.halfEdges[i];
		const auto& e2 = vor.halfEdges[i + 1];
		if (e1.vertexFrom != -1 && e2.vertexFrom != -1)
		{
			edges[edgeCounter] = e1.vertexFrom;
			edges[edgeCounter + edgeCount] = e2.vertexFrom;
			++edgeCounter;
		}
		else // Infinite or half-infinite edge.
		{
			const auto s1 = e1.face;
			const auto s2 = e2.face;
			assert(s1 != s2);
			// a * y + b * x + c = 0
			const auto a = sites[s2].y - sites[s1].y;
			const auto b = sites[s2].x - sites[s1].x;
			const auto c = -b * (sites[s1].x + sites[s2].x) / 2 - a * (sites[s1].y + sites[s2].y) / 2;

			if (e1.vertexFrom != -1 || e2.vertexFrom != -1)
			{
				const auto& [e, eTwinInd] = e1.vertexFrom == -1 ? make_tuple(e2, i) : make_tuple(e1, i + 1);
				// prev = twin->next->twin->next->twin
				const auto eTwinNextInd = vor.halfEdges[eTwinInd].next;
				const auto eTwinNextTwinInd = eTwinNextInd % 2 == 0 ? eTwinNextInd + 1 : eTwinNextInd - 1;
				const auto ePrevTwinInd = vor.halfEdges[eTwinNextTwinInd].next;
				// sOpposite - site opposite to the ray.
				const auto sOpposite = vor.halfEdges[ePrevTwinInd].face;
				assert(s1 != sOpposite && s2 != sOpposite);

				infEdgeXs[infEdgeCounter] = vor.vertices[e.vertexFrom].p.x;
				infEdgeYs[infEdgeCounter] = vor.vertices[e.vertexFrom].p.y;

				if (isCloseToZero(a))
				{
					infEdgeXs[infEdgeCounter + infEdgeCount] = -c / b;
					infEdgeYs[infEdgeCounter + infEdgeCount] = sites[sOpposite].y < sites[s1].y ? drawMaxY : drawMinY;
				}
				else
				{
					const auto isDirLeft = isClose(sites[s1].x, sites[s2].x)
						? sites[s1].x < sites[sOpposite].x
						: [a, b, &s1 = sites[s1], &sOpposite = sites[sOpposite]]()
					{
						const auto perpA = -b;
						const auto perpB = a;
						const auto perpC = -perpA * s1.y - perpB * s1.x;
						const auto distToOpposite = perpA * sOpposite.y + perpB * sOpposite.x + perpC;
						const auto distToPointToRight = perpA * s1.y + perpB * (s1.x + 1) + perpC;
						return distToPointToRight * distToOpposite > 0;
					}();

					infEdgeXs[infEdgeCounter + infEdgeCount] = isDirLeft ? drawMinX : drawMaxX;
					infEdgeYs[infEdgeCounter + infEdgeCount] = -(b * infEdgeXs[infEdgeCounter + infEdgeCount] + c) / a;
				}

				++infEdgeCounter;
			}
			else // Infinite edge.
			{
				if (isCloseToZero(a))
				{
					doubleInfEdgeXs[doubleInfEdgeCounter] = -c / b;
					doubleInfEdgeYs[doubleInfEdgeCounter] = drawMinY;
					doubleInfEdgeXs[doubleInfEdgeCounter + doubleInfEdgeCount] = -c / b;
					doubleInfEdgeYs[doubleInfEdgeCounter + doubleInfEdgeCount] = drawMaxY;
				}
				else
				{
					doubleInfEdgeXs[doubleInfEdgeCounter] = drawMinX;
					doubleInfEdgeYs[doubleInfEdgeCounter] = -(b * drawMinX + c) / a;
					doubleInfEdgeXs[doubleInfEdgeCounter + doubleInfEdgeCount] = drawMaxX;
					doubleInfEdgeYs[doubleInfEdgeCounter + doubleInfEdgeCount] = -(b * drawMaxX + c) / a;
				}

				++doubleInfEdgeCounter;
			}
		}
	}

	vector<size_t> delaunayEdges;
	delaunayEdges.resize(vor.halfEdges.size());
	for (size_t i{ 0 }; i != vor.halfEdges.size(); i += 2)
	{
		// vor.edges[i + 1] is a twin edge of an vor.edges[i].
		delaunayEdges[i / 2] = vor.halfEdges[i].face;
		delaunayEdges[(i + vor.halfEdges.size()) / 2] = vor.halfEdges[i + 1].face;
	}

	vector<double> siteXs;
	vector<double> siteYs;
	for(const auto& p : sites)
	{
		siteXs.push_back(p.x);
		siteYs.push_back(p.y);
	}

	try
	{
		py::scoped_interpreter interpreter_guard{};

		const auto np = py::module_::import("numpy");
		const auto plt = py::module_::import("matplotlib.pyplot");

		const auto pySitesXs = to_pyarray(move(siteXs));
		const auto pySitesYs = to_pyarray(move(siteYs));
		const auto pyVertexXs = to_pyarray(move(vertexXs));
		const auto pyVertexYs = to_pyarray(move(vertexYs));
		const auto pyEdges = to_pyarray(move(edges), vector<size_t>{2, edgeCount});
		const auto pyDelaunayEdges = to_pyarray(move(delaunayEdges), vector<size_t>{2, vor.halfEdges.size() / 2});

		const auto fig = plt.attr("figure")();
		const auto ax = fig.attr("add_subplot")(111);

		ax.attr("scatter")(pySitesXs, pySitesYs, "c"_a = "r");
		ax.attr("scatter")(pyVertexXs, pyVertexYs, "c"_a = "b");
		ax.attr("plot")(pyVertexXs[pyEdges], pyVertexYs[pyEdges], "y-");
		ax.attr("plot")(to_pyarray(move(infEdgeXs), vector<size_t>{2, infEdgeCount}), to_pyarray(move(infEdgeYs), vector<size_t>{2, infEdgeCount}), "y-");
		ax.attr("plot")(to_pyarray(move(doubleInfEdgeXs), vector<size_t>{2, doubleInfEdgeCount}), to_pyarray(move(doubleInfEdgeYs), vector<size_t>{2, doubleInfEdgeCount}), "y-");
		ax.attr("plot")(pySitesXs[pyDelaunayEdges], pySitesYs[pyDelaunayEdges], "r-");
		ax.attr("set_aspect")(1);
		plt.attr("xlim")(drawMinX, drawMaxX);
		plt.attr("ylim")(drawMinY, drawMaxY);
		plt.attr("show")();
	}
	catch (std::exception e)
	{
		std::cout << e.what() << "\n";
	}

	return 0;
}
