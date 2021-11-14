#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>
#include <cassert>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>

using namespace std;
namespace py = pybind11;
using namespace py::literals;

bool approximatelyEqual(double a, double b, double epsilon = numeric_limits<double>::epsilon())
{
	return abs(a - b) <= ((abs(a) < abs(b) ? abs(b) : abs(a)) * epsilon);
}

bool essentiallyEqual(double a, double b, double epsilon = numeric_limits<double>::epsilon())
{
	return abs(a - b) <= ((abs(a) > abs(b) ? abs(b) : abs(a)) * epsilon);
}

bool definitelyGreaterThan(double a, double b, double epsilon = numeric_limits<double>::epsilon())
{
	return (a - b) > ((abs(a) < abs(b) ? abs(b) : abs(a)) * epsilon);
}

bool definitelyLessThan(double a, double b, double epsilon = numeric_limits<double>::epsilon())
{
	return (b - a) > ((abs(a) < abs(b) ? abs(b) : abs(a)) * epsilon);
}

bool isClose(double a, double b, double maxRelDiff = 1e-9, double maxAbsDiff = 1e-50)
{
	const auto diff = abs(a - b);
	return diff <= maxRelDiff * abs(a)
		|| diff <= maxRelDiff * abs(b)
		|| diff <= maxAbsDiff;
}

bool isCloseToZero(double a, double maxAbsDiff = 1e-50)
{
	return isClose(a, 0.0, 0.0, maxAbsDiff);
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
		size_t edge;
	};
	struct Face
	{
		size_t edge;
	};
	struct Edge
	{
		ptrdiff_t vertexFrom = -1; // -1 for edges that starts from infinite.
		
		size_t face;

		ptrdiff_t next = -1; // -1 if there is no next (infinite edge).
		ptrdiff_t prev = -1;

		// NOTE: twin edges located in sequence in edges array, i.e., edges[2 * i] and edges[2 * i + 1] are twins.
	};
	vector<Edge> edges; // up to n^2 elements?
	vector<Vertex> vertices;
	vector<Face> faces;
};

enum class ParabolasIntersectionDirection
{
	left,
	right,
	down
};

// Returns x-coordinate of an intersection of the arc1 from the left and arc2 from the right and whether it is left intersection (with lower x).
tuple<double, ParabolasIntersectionDirection> parabolasIntersectionX(const double sweepLineY, const Point& site1, const Point& site2)
{
	const auto sx1 = site1.x;
	const auto sx2 = site2.x;
	const auto sy1 = site1.y;
	const auto sy2 = site2.y;
	if (isClose(sy1, sy2))
	{
		return make_tuple((sx1 + sx2) / 2, ParabolasIntersectionDirection::down);
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
		? make_tuple((mb - D) / pDiff, changeSign ? ParabolasIntersectionDirection::right : ParabolasIntersectionDirection::left)
		: make_tuple((mb + D) / pDiff, changeSign ? ParabolasIntersectionDirection::left : ParabolasIntersectionDirection::right);
}

double parabolaY(const double sweepLineY, const Point& site, const double x)
{
	const auto p = site.y - sweepLineY;
	assert(!isCloseToZero(p));
	return x * x / (2 * p) - (site.x * x) / p + (site.x * site.x) / (2 * p) + p / 2 + sweepLineY;
}

struct BeachLine
{
	// Implemented as AVL tree.

	struct Node
	{
		Node* parent;
		Node* left;
		Node* right;
		size_t p;
		size_t q;
		ptrdiff_t edgepq = -1;
		ptrdiff_t circleEventId; // -1 if no circle event is connected to this leaf.
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
	const std::vector<Point>& points;

	constexpr BeachLine(const std::vector<Point>& points, std::vector<size_t> pointsWithTheBiggestY, DoublyConnectedEdgeList& dcel)
		: points{ points }
		, root{ nullptr }
	{
		assert(pointsWithTheBiggestY.size() >= 1);
		std::sort(
			std::begin(pointsWithTheBiggestY), std::end(pointsWithTheBiggestY),
			[&points](const size_t lhs, const size_t rhs)
			{
				return points[lhs].x < points[rhs].x;
			});
		root = init(std::cbegin(pointsWithTheBiggestY), std::cend(pointsWithTheBiggestY), dcel);
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
			.edgepq = static_cast<ptrdiff_t>(dcel.edges.size()),
			.circleEventId = -1,
			.balance = static_cast<signed char>(mid * 2 == dist ? 0 : 1)
		};
		n->left->parent = n;
		n->right->parent = n;
		dcel.edges.push_back({ .face = s1 });
		dcel.edges.push_back({ .face = s2 });
		dcel.faces[s1] = { .edge = dcel.edges.size() - 2 };
		dcel.faces[s2] = { .edge = dcel.edges.size() - 1 };
		return n;
	}

	constexpr Node* findRegion(const size_t site) const
	{
		return findRegionFrom(root, site, points[site].y);
	}

	constexpr Node* findRegionFrom(Node* node, const size_t site, const double sweepLinePos) const
	{
		assert(node != nullptr);
		if (node->isLeaf())
			return node;

		const auto intersectionX = get<0>(parabolasIntersectionX(sweepLinePos, points[node->p], points[node->q]));

		// If intersection's x-coordinate equals to site's x-coordinate, then new circle event will be created and instantly processed
		// (piece of left arc with is created and removed instantly).
		// Then the same will happen to the right arc of this intersection.
		const auto nextNode = points[site].x <= intersectionX ? node->left : node->right;
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
		newSubTree->left->edgepq = -1;
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

	constexpr PriorityQueue(const vector<Point>& points)
		: storage{ new Event[points.size() * 2] }
		, ids{ new size_t[points.size() * 2] }
		, size{ points.size() }
		, newId{ points.size() }
	{
		idToInd.reserve(points.size());
		for (size_t i{ 0 }; i != points.size(); ++i)
		{
			storage[i] = Event{ .y = points[i].y, .type = Event::Type::site };
			ids[i] = i;
			idToInd.push_back(i);
		}
		for (size_t i{ points.size() / 2 + 1 }; i != 0; --i)
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

bool isConvergent(const double y, const Point& siteLeft, const Point& siteCenter, const Point& siteRight)
{
	const auto [leftIntersectionX, leftIntersectionDir] = parabolasIntersectionX(y, siteLeft, siteCenter);
	const auto [rightIntersectionX, rightIntersectionDir] = parabolasIntersectionX(y, siteCenter, siteRight);

	if (leftIntersectionDir == ParabolasIntersectionDirection::down && rightIntersectionDir == ParabolasIntersectionDirection::down)
	{
		return false;
	}

	if (leftIntersectionX == rightIntersectionX && siteCenter.y < siteLeft.y && siteCenter.y < siteRight.y)
	{
		// If arc has 0 width (just appears) and is growing.
		return false;
	}

	// a * y + b * x + c = 0
	const auto a1 = (siteCenter.y - siteLeft.y);
	const auto b1 = (siteCenter.x - siteLeft.x);
	const auto c1 = -b1 * (siteLeft.x + siteCenter.x) / 2 - a1 * (siteLeft.y + siteCenter.y) / 2;

	const auto a2 = (siteRight.y - siteCenter.y);
	const auto b2 = (siteRight.x - siteCenter.x);
	const auto c2 = -b2 * (siteCenter.x + siteRight.x) / 2 - a2 * (siteCenter.y + siteRight.y) / 2;

	assert(!isCloseToZero(a1) || !isCloseToZero(a2));

	const auto denom = a2 * b1 - a1 * b2;
	if (isCloseToZero(denom))
	{
		// Lines are parallel, there is no intersection.
		return false;
	}
	const auto intersectionX = (a1 * c2 - a2 * c1) / denom;

	switch (leftIntersectionDir)
	{
		break; case ParabolasIntersectionDirection::left:
		{
			if (leftIntersectionX < intersectionX)
				return false;
		}
		break; case ParabolasIntersectionDirection::right:
		{
			if (leftIntersectionX > intersectionX)
				return false;
		}
		break; case ParabolasIntersectionDirection::down:
		{
			if (!isClose(leftIntersectionX, intersectionX))
				return false;
		}
	}

	switch (rightIntersectionDir)
	{
		break; case ParabolasIntersectionDirection::left:
		{
			if (intersectionX > rightIntersectionX)
				return false;
		}
		break; case ParabolasIntersectionDirection::right:
		{
			if (intersectionX < rightIntersectionX)
				return false;
		}
		break; case ParabolasIntersectionDirection::down:
		{
			if (!isClose(rightIntersectionX, intersectionX))
				return false;
		}
	}

	// TODO: if this assert is false, then return false in this case.
	assert(isClose(y, get<0>(circleBottomPoint(siteLeft, siteCenter, siteRight))) || y > get<0>(circleBottomPoint(siteLeft, siteCenter, siteRight)));

	return true;
}

DoublyConnectedEdgeList fortune(const vector<Point>& points)
{
	auto dcel = DoublyConnectedEdgeList{};

	dcel.faces.resize(points.size());

	auto queue = PriorityQueue{ points };

	if (queue.empty())
	{
		return dcel;
	}

	auto pointsWithTheBiggestY = vector<size_t>{};
	do
	{
		pointsWithTheBiggestY.push_back(get<1>(queue.pop()));
	} while (!queue.empty() && isClose(queue.storage[0].y, points[pointsWithTheBiggestY.back()].y));
	
	auto beachLine = BeachLine{ points, std::move(pointsWithTheBiggestY), dcel };

	const auto createCircleEvents = [&points, &queue, &beachLine]
	(const double y,
		BeachLine::Node* left, BeachLine::Node* leftIntersection, BeachLine::Node* innerLeft,
		BeachLine::Node* innerRight, BeachLine::Node* rightIntersection, BeachLine::Node* right)
	{
		if(const auto leftleftIntersection = get<0>(beachLine.findIntersectionWithLeftLeaf(left)))
		{
			if(isConvergent(y, points[leftleftIntersection->p], points[leftleftIntersection->q], points[innerLeft->p]))
			{
				//const auto leftleftSite = leftleftIntersection->p == left->p ? leftleftIntersection->q : leftleftIntersection->p;
				const auto leftleftSite = leftleftIntersection->p;
				assert(leftleftSite != left->p && leftleftSite != innerLeft->p && left->p != innerLeft->p);
				const auto [bottom, center] = circleBottomPoint(points[leftleftSite], points[left->p], points[innerLeft->p]);
				left->circleEventId = queue.insertCircleEvent(bottom, center, left);
			}
		}
		if (const auto rightrightIntersection = get<0>(beachLine.findIntersectionWithRightLeaf(right)))
		{
			if(isConvergent(y, points[innerRight->p], points[rightrightIntersection->p], points[rightrightIntersection->q]))
			{
				//const auto rightrightSite = rightrightIntersection->p == right->p ? rightrightIntersection->q : rightrightIntersection->p;
				const auto rightrightSite = rightrightIntersection->q;
				assert(rightrightSite != right->p && rightrightSite != innerRight->p && right->p != innerRight->p);
				const auto [bottom, center] = circleBottomPoint(points[rightrightSite], points[right->p], points[innerRight->p]);
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
			dcel.edges.push_back({ .face = central->p });
			dcel.edges.push_back({ .face = left->p });
			dcel.faces[central->p] = { .edge = dcel.edges.size() - 2 };
			dcel.faces[left->p] = { .edge = dcel.edges.size() - 1 };
			centralRight->edgepq = leftCentral->edgepq = dcel.edges.size() - 2;
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

			assert(leftIntersection->edgepq != -1);
			assert(rightIntersection->edgepq != -1);

			const auto bads = arcToRemove->p;
			const auto leftIntersectionEdgepq = leftIntersection->edgepq;
			const auto rightIntersectionEdgepq = rightIntersection->edgepq;
			const auto intersection = beachLine.removeArc(arcToRemove);

			const auto s1 = left->p;
			const auto s2 = right->p;

			dcel.edges.push_back({ .face = s1 });
			dcel.edges.push_back({ .face = s2 });

			const auto intersectionEdgepq = intersection->edgepq = dcel.edges.size() - 2;

			const auto setVertexFrom = [&dcel](const ptrdiff_t edgepq, const size_t s)
			{
				if (dcel.edges[edgepq].face == s)
				{
					dcel.edges[edgepq].vertexFrom = dcel.vertices.size();
				}
				else
				{
					dcel.edges[edgepq + 1].vertexFrom = dcel.vertices.size();
				}
			};
			
			setVertexFrom(leftIntersectionEdgepq, s1);
			setVertexFrom(rightIntersectionEdgepq, bads);
			setVertexFrom(intersectionEdgepq, s2);

			dcel.vertices.push_back({
				.p = ev.center,
				.edge = dcel.edges[intersectionEdgepq].face == s2 ? static_cast<size_t>(intersectionEdgepq) : static_cast<size_t>(intersectionEdgepq + 1)
			});

			// TODO: if this assertion about left orientation of a triangle (s1, bads, s2) is not true,
			// then put body of setNextAndPrev in if block, copy it to else block with swapping next and prev in lhs.
			assert((points[bads].x - points[s1].x) * (points[s2].y - points[s1].y) - (points[bads].y - points[s1].y) * (points[s2].x - points[s1].x) < 0);

			const auto setNextAndPrev = [&edges = dcel.edges](const ptrdiff_t centralEdgepq, const ptrdiff_t leftEdgepq, const ptrdiff_t rightEdgepq, const size_t sLeft, const size_t sRight)
			{
				if (edges[centralEdgepq].face == sLeft)
				{
					edges[centralEdgepq].next = edges[leftEdgepq].face == sLeft ? leftEdgepq : leftEdgepq + 1;
					edges[centralEdgepq + 1].prev = edges[rightEdgepq].face == sRight ? rightEdgepq : rightEdgepq + 1;
				}
				else
				{
					edges[centralEdgepq + 1].next = edges[leftEdgepq].face == sLeft ? leftEdgepq : leftEdgepq + 1;
					edges[centralEdgepq].prev = edges[rightEdgepq].face == sRight ? rightEdgepq : rightEdgepq + 1;
				}
			};

			setNextAndPrev(intersectionEdgepq, leftIntersectionEdgepq, rightIntersectionEdgepq, s1, s2);
			setNextAndPrev(leftIntersectionEdgepq, rightIntersectionEdgepq, intersectionEdgepq, bads, s1);
			setNextAndPrev(rightIntersectionEdgepq, intersectionEdgepq, leftIntersectionEdgepq, s2, bads);

			createCircleEvents(ev.y, left, intersection, right, left, intersection, right);
		}
		}
	}

	return dcel;
}

int main()
{
	//const auto points = vector<Point>{ {0, 10}, {1, 9}, {5, 8}, {3, 4}, {4, 5}, {1,-1}, {5,-2}, {-5,-5}, {-10,-6}, {-9, 2}, {-11,7}, {-3, 0}, {-2,6}, {-11, 11}, {-11, 7.5} };
	//const auto points = vector<Point>{ {1, 2}, {0, 1}, {0, 0}, {0, -1}, {0, -2}, {0, -3}, {0, -4} };
	//const auto points = vector<Point>{ {4, 0}, {0, 8}, {8, 2}, {7, 9} };
	//const auto points = vector<Point>{ {1, 9}, {5, 8}, {3, 4}, {4, 5}, {1,-1}, {5,-2}, {-10,-6}, {-11,7}, {-2,6}, {-11, 11} };
	//const auto points = vector<Point>{ {-1, 1}, {1, 1}, {0, 0}, {3, 1} };
	// auto points = vector<Point>{ {0, 2}, {0, 1}, {0, 0}, {0, -1}, {0, -2} };
	//const auto points = vector<Point>{ {-1, 1}, {1, 1}, {3, 1}, {0, 4}, {2, 4}, {-1, 7}, {1, 7}, {5, 7} };
	const auto points = vector<Point>{ {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {1, 0}, {2, 0}, {3, 0}, {1, 4}, {2, 4}, {3, 4} };
	const auto vor = fortune(points);
	const auto minMaxXY = [](const tuple<double, double, double, double>& accum, const Point& p)
	{
		return make_tuple(min(get<0>(accum), p.x), max(get<1>(accum), p.x), min(get<2>(accum), p.y), max(get<3>(accum), p.y));
	};
	const auto pointsMinMax = points.size() == 0
		? make_tuple(0.0, 0.0, 0.0, 0.0)
		: reduce(
			cbegin(points), cend(points),
			make_tuple(numeric_limits<double>::max(), numeric_limits<double>::min(), numeric_limits<double>::max(), numeric_limits<double>::min()),
			minMaxXY
		);
	constexpr auto extendDrawRegionForAllVertices = true;
	const auto [minX, maxX, minY, maxY] = !extendDrawRegionForAllVertices
		? pointsMinMax
		: transform_reduce(
			cbegin(vor.vertices), cend(vor.vertices),
			pointsMinMax,
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

	vector<size_t> edgeAs;
	vector<size_t> edgeBs;
	vector<size_t> infEdgeAs;
	vector<double> infEdgeBXs;
	vector<double> infEdgeBYs;
	vector<double> doubleInfEdgeAXs;
	vector<double> doubleInfEdgeAYs;
	vector<double> doubleInfEdgeBXs;
	vector<double> doubleInfEdgeBYs;
	for (size_t i{ 0 }; i < vor.edges.size(); i += 2)
	{
		const auto& e1 = vor.edges[i];
		const auto& e2 = vor.edges[i + 1];
		if (e1.vertexFrom != -1 && e2.vertexFrom != -1)
		{
			edgeAs.push_back(e1.vertexFrom);
			edgeBs.push_back(e2.vertexFrom);
		}
		else if (e1.vertexFrom != -1 || e2.vertexFrom != -1)
		{
			const auto& [e, eTwinInd] = e1.vertexFrom == -1 ? make_tuple(e2, i) : make_tuple(e1, i + 1);
			// s1, s2 - sLeft, sRight.
			const auto s1 = e.face;
			const auto s2 = vor.edges[eTwinInd].face;
			const auto ePrevTwinInd = e.prev % 2 == 0 ? e.prev + 1 : e.prev - 1;
			// bads - site opposite to the ray, sOpposite.
			const auto bads = vor.edges[ePrevTwinInd].face;
			assert(s1 != s2 && s1 != bads && s2 != bads);

			infEdgeAs.push_back(e.vertexFrom);

			// a * y + b * x + c = 0
			const auto a = points[s2].y - points[s1].y;
			const auto b = points[s2].x - points[s1].x;
			const auto c = -b * (points[s1].x + points[s2].x) / 2 - a * (points[s1].y + points[s2].y) / 2;

			if (isCloseToZero(a))
			{
				infEdgeBXs.push_back(-c / b);
				infEdgeBYs.push_back(points[bads].y < points[s1].y ? drawMaxY : drawMinY);
			}
			else
			{
				const auto calcIsDirLeft = [](const Point& s1, const Point& s2, const Point& bads)
				{
					if (!isClose(s1.x, s2.x))
					{
						// ax + y + c = 0
						const auto a = -(s2.y - s1.y) / (s2.x - s1.x);
						const auto c = -(s2.y + a * s2.x);

						const auto dist = (a * bads.x + bads.y + c);

						const auto middleX = (s2.x + s1.x) / 2;
						const auto middleY = (s2.y + s1.y) / 2;

						return (a * (middleX + 1) + middleY + c) * dist > 0;
					}

					return s1.x < bads.x;
				};
				const auto isDirLeft = calcIsDirLeft(points[s1], points[s2], points[bads]);

				// Ray: y = m * x + f
				const auto m = (points[s1].x - points[s2].x) / (points[s2].y - points[s1].y);
				const auto f = -m * (points[s1].x + points[s2].x) / 2 + (points[s1].y + points[s2].y) / 2;
				infEdgeBXs.push_back(isDirLeft ? drawMinX : drawMaxX);
				infEdgeBYs.push_back(m * infEdgeBXs.back() + f);
			}
		}
		else
		{
			const auto s1 = e1.face;
			const auto s2 = e2.face;
			assert(s1 != s2);
			const auto a = points[s2].y - points[s1].y;
			const auto b = points[s2].x - points[s1].x;
			const auto c = -b * (points[s1].x + points[s2].x) / 2 - a * (points[s1].y + points[s2].y) / 2;
			/*const auto m = (points[s1].x - points[s2].x) / (points[s2].y - points[s1].y);
			const auto f = -m * (points[s1].x + points[s2].x) / 2 + (points[s1].y + points[s2].y) / 2;*/
			if (isCloseToZero(a))
			{
				doubleInfEdgeAXs.push_back(-c / b);
				doubleInfEdgeAYs.push_back(drawMinY);
				doubleInfEdgeBXs.push_back(-c / b);
				doubleInfEdgeBYs.push_back(drawMaxY);
			}
			else
			{
				doubleInfEdgeAXs.push_back(drawMinX);
				//doubleInfEdgeAYs.push_back(m * drawMinX + f);
				doubleInfEdgeAYs.push_back(-(b * drawMinX + c) / a);
				doubleInfEdgeBXs.push_back(drawMaxX);
				//doubleInfEdgeBYs.push_back(m * drawMaxX + f);
				doubleInfEdgeBYs.push_back(-(b * drawMaxX + c) / a);
			}
		}
	}

	vector<size_t> delaunayEdgeAs;
	vector<size_t> delaunayEdgeBs;
	for (size_t i{ 0 }; i != vor.edges.size(); i += 2)
	{
		// vor.edges[i + 1] is a twin edge of an vor.edges[i].
		delaunayEdgeAs.push_back(vor.edges[i].face);
		delaunayEdgeBs.push_back(vor.edges[i + 1].face);
	}

	vector<double> pointXs;
	vector<double> pointYs;
	for(const auto& p : points)
	{
		pointXs.push_back(p.x);
		pointYs.push_back(p.y);
	}

	try
	{
		py::scoped_interpreter interpreter_guard{};

		py::dict locals{
			"drawMinX"_a = drawMinX, "drawMaxX"_a = drawMaxX, "drawMinY"_a = drawMinY, "drawMaxY"_a = drawMaxY,
			"vertexXs"_a = vertexXs, "vertexYs"_a = vertexYs,
			"edgeAs"_a = edgeAs, "edgeBs"_a = edgeBs,
			"infEdgeAs"_a = infEdgeAs, "infEdgeBXs"_a = infEdgeBXs, "infEdgeBYs"_a = infEdgeBYs,
			"doubleInfEdgeAXs"_a = doubleInfEdgeAXs, "doubleInfEdgeAYs"_a = doubleInfEdgeAYs, "doubleInfEdgeBXs"_a = doubleInfEdgeBXs, "doubleInfEdgeBYs"_a = doubleInfEdgeBYs,
			"delaunayEdgeAs"_a = delaunayEdgeAs, "delaunayEdgeBs"_a = delaunayEdgeBs,
			"pointXs"_a = pointXs, "pointYs"_a = pointYs
		};

		py::exec(R"(
			import numpy as np
			import matplotlib.pyplot as plt

			vertexXs = np.array(vertexXs)
			vertexYs = np.array(vertexYs)

			edges = np.zeros((2, len(edgeAs)), dtype = np.uint64)
			edges[0, :] = np.array(edgeAs)
			edges[1, :] = np.array(edgeBs)

			infEdgeXs = np.zeros((2, len(infEdgeAs)))
			infEdgeXs[0, :] = vertexXs[infEdgeAs]
			infEdgeXs[1, :] = np.array(infEdgeBXs)
			infEdgeYs = np.zeros((2, len(infEdgeAs)))
			infEdgeYs[0, :] = vertexYs[infEdgeAs]
			infEdgeYs[1, :] = np.array(infEdgeBYs)

			doubleInfEdgeXs = np.zeros((2, len(doubleInfEdgeAXs)))
			doubleInfEdgeXs[0, :] = np.array(doubleInfEdgeAXs)
			doubleInfEdgeXs[1, :] = np.array(doubleInfEdgeBXs)
			doubleInfEdgeYs = np.zeros((2, len(doubleInfEdgeAYs)))
			doubleInfEdgeYs[0, :] = np.array(doubleInfEdgeAYs)
			doubleInfEdgeYs[1, :] = np.array(doubleInfEdgeBYs)

			delaunayEdges = np.zeros((2, len(delaunayEdgeAs)), dtype = np.uint64)
			delaunayEdges[0, :] = np.array(delaunayEdgeAs)
			delaunayEdges[1, :] = np.array(delaunayEdgeBs)

			pointXs = np.array(pointXs)
			pointYs = np.array(pointYs)
			
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.scatter(vertexXs, vertexYs, c = 'b')
			ax.plot(vertexXs[edges], vertexYs[edges], 'y-')
			ax.plot(infEdgeXs, infEdgeYs, 'y-')
			ax.plot(doubleInfEdgeXs, doubleInfEdgeYs, 'y-')
			ax.plot(pointXs[delaunayEdges], pointYs[delaunayEdges], 'r-')
			ax.scatter(pointXs, pointYs, c = 'r')
			ax.set_aspect(1)
			plt.xlim([drawMinX, drawMaxX])
			plt.ylim([drawMinY, drawMaxY])
			plt.show()
			)",
			py::globals(), locals);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << "\n";
	}

	return 0;
}
