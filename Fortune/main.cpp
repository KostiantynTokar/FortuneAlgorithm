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

bool isClose(double a, double b, double maxRelDiff = 1e-9, double maxAbsDiff = 0.0)
{
	const auto diff = abs(a - b);
	return diff <= maxRelDiff * abs(a)
		|| diff <= maxRelDiff * abs(b)
		|| diff <= maxAbsDiff;
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

// Returns x-coordinate of left intersection if site1.x <= site2.x and x-coordinate of right intersection otherwise.
double parabolasIntersectionX(const double sweepLineY, const Point& site1, const Point& site2)
{
	const auto sx1 = site1.x;
	const auto sx2 = site2.x;
	const auto sy1 = site1.y;
	const auto sy2 = site2.y;
	const auto p1 = site1.y - sweepLineY;
	const auto p2 = site2.y - sweepLineY;

	const auto syDiff = sy2 - sy1;
	//const auto D = sqrt(p1 * p2) * hypot(sx1 - sx2, sy1 - sy2);
	const auto mb = sx1 * p2 - sx2 * p1;

	const auto syDiffSign = syDiff >= 0 ? 1.0 : -1.0;
	//return syDiffSign * sx1 <= syDiffSign * sx2 ? (mb - D) / syDiff : (mb + D) / syDiff;

	const auto pDiff = p2 - p1;
	const auto sxDiff = sx2 - sx1;
	const auto D = sqrt(p1 * p2 * (sxDiff * sxDiff + pDiff * pDiff));
	return syDiffSign * sx1 <= syDiffSign * sx2 ? (mb - D) / pDiff : (mb + D) / pDiff;
}

constexpr double parabolaY(const double sweepLineY, const Point& site, const double x)
{
	const auto p = site.y - sweepLineY;
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

	constexpr BeachLine(const std::vector<Point>& points)
		: points{ points }
		, root{ nullptr }
	{
	}

	~BeachLine()
	{
		deleteRecursive(root);
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

		const auto x = parabolasIntersectionX(sweepLinePos, points[node->p], points[node->q]);

		const auto nextNode = points[site].x <= x ? node->left : node->right;
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
		if (empty())
		{
			root = new Node{ .parent = nullptr, .left = nullptr, .right = nullptr, .p = site, .q = 0, .circleEventId = -1, .balance = 0 };
			return { -1, nullptr, nullptr, root, nullptr, nullptr };
		}
		auto regionNode = findRegion(site);
		const auto intersectedArc = regionNode->p;
		const auto eventId = regionNode->circleEventId;
		auto regionNodeParent = regionNode->parent;

		size_t leftInd;
		size_t rightInd;
		if (points[intersectedArc].x < points[site].x)
		{
			leftInd = intersectedArc;
			rightInd = site;
		}
		else
		{
			leftInd = site;
			rightInd = intersectedArc;
		}

		auto newSubTree = new Node{ .parent = regionNodeParent, .left = nullptr, .right = nullptr, .p = leftInd, .q = rightInd, .circleEventId = -1, .balance = 0 };
		newSubTree->left = regionNode;
		newSubTree->left->parent = newSubTree;
		newSubTree->left->circleEventId = -1;
		newSubTree->left->edgepq = -1;
		newSubTree->right = new Node{ .parent = newSubTree, .left = nullptr, .right = nullptr, .p = rightInd, .q = leftInd, .circleEventId = -1, .balance = 0 };
		const auto redLeaf = newSubTree->right; // Rebalance, then add to this node 2 children.

		auto res = NewArcInfo{
			.id = eventId,
			.left = newSubTree->left,
			.leftCentralIntersection = newSubTree,
			.centralRightIntersection = newSubTree->right,
		};
		
		replaceNode(regionNode, newSubTree, regionNodeParent);

		auto child = newSubTree;
		bool needRebalance = true;
		while (regionNodeParent != nullptr && (needRebalance || child->balance != 0))
		{
			if (regionNodeParent->left == child)
				--regionNodeParent->balance;
			else
				++regionNodeParent->balance;
			assert(-2 <= regionNodeParent->balance && regionNodeParent->balance <= 2);
			child = regionNodeParent;
			regionNodeParent = regionNodeParent->parent;
			needRebalance = rebalanceSubTree(child, regionNodeParent);
		}

		redLeaf->left = new Node{ .parent = redLeaf, .left = nullptr, .right = nullptr, .p = site, .q = 0, .circleEventId = -1, .balance = 0 };
		redLeaf->right = new Node{ .parent = redLeaf, .left = nullptr, .right = nullptr, .p = intersectedArc, .q = 0, .circleEventId = -1, .balance = 0 };

		res.central = redLeaf->left;
		res.right = redLeaf->right;

		child = redLeaf;
		regionNodeParent = child->parent;
		needRebalance = true;
		while (regionNodeParent != nullptr && (needRebalance || child->balance != 0))
		{
			if (regionNodeParent->left == child)
				--regionNodeParent->balance;
			else
				++regionNodeParent->balance;
			assert(-2 <= regionNodeParent->balance && regionNodeParent->balance <= 2);
			child = regionNodeParent;
			regionNodeParent = regionNodeParent->parent;
			needRebalance = rebalanceSubTree(child, regionNodeParent);
		}

		return res;
	}

	// Returns node with new intersection (between leaf to the left of node and leaf to the right of node).
	constexpr Node* removeArc(Node* node)
	{
		assert(node->isLeaf());
		auto [leftIntersection, leftHeight] = findIntersectionWithLeftLeaf(node);
		auto [rightIntersection, rightHeight] = findIntersectionWithRightLeaf(node);
		assert(leftHeight == 1 || rightHeight == 1);

		auto [higherNode, lowerNode] = leftHeight > rightHeight ? make_tuple(leftIntersection, rightIntersection) : make_tuple(rightIntersection, leftIntersection);
		auto s1 = higherNode->p == node->p ? higherNode->q : higherNode->p;
		auto s2 = lowerNode->p == node->p ? lowerNode->q : lowerNode->p;
		bool isNewIntersectionLeft;
		if (points[s1].y < points[s2].y)
			isNewIntersectionLeft = points[higherNode->p].x < points[higherNode->q].x;
		else
			isNewIntersectionLeft = points[lowerNode->p].x < points[lowerNode->q].x;
		if (points[s1].x > points[s2].x)
		{
			auto tmp = s1;
			s1 = s2;
			s2 = tmp;
		}

		if (isNewIntersectionLeft)
		{
			higherNode->p = s1;
			higherNode->q = s2;
		}
		else
		{
			higherNode->p = s2;
			higherNode->q = s1;
		}

		auto insteadLower = lowerNode->left == node ? lowerNode->right : lowerNode->left;
		auto aboveLower = lowerNode->parent;
		if (lowerNode->isLeft())
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
		auto needRebalance = rebalanceSubTree(aboveLower, parent);

		while (parent != nullptr && (needRebalance || aboveLower->balance == 0))
		{
			// Height of subtree rooted at aboveLower is decreased.
			if (parent->left == aboveLower)
				++parent->balance;
			else
				--parent->balance;
			aboveLower = parent;
			parent = parent->parent;
			needRebalance = rebalanceSubTree(aboveLower, parent);
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

	constexpr bool rebalanceSubTree(Node*& node, Node* parent)
	{
		if (node->balance == 2)
		{
			auto child = node->right;
			if (child->balance >= 0)
			{
				// Right right rotation.
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
				// Right left rotation.
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
			return true;
		}
		else if (node->balance == -2)
		{
			auto child = node->left;
			if (child->balance <= 0)
			{
				// Left left rotation.
				replaceNode(node, child, parent);
				node->parent = child;
				node->left = child->right;
				child->right->parent = node;
				child->right = node;

				node->balance = 1 + child->balance;
				++child->balance;

				node = child;
			}
			else
			{
				// Left right rotation.
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
			return true;
		}
		return false;
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
	const auto ma = (b.y - a.y) / (b.x - a.x);
	const auto mb = (c.y - b.y) / (c.x - b.x);
	const auto xc = (ma * mb * (a.y - c.y) + mb * (a.x + b.x) - ma * (b.x + c.x)) / (2 * (mb - ma));
	const auto yc = ((a.x + b.x) / 2 - xc) / ma + (a.y + b.y) / 2;
	const auto r = hypot(a.x - xc, a.y - yc);
	return make_tuple(yc - r, Point{ xc, yc });
}

bool isConvergent(const double y, const Point& a1, const Point& a2, const Point& b1, const Point& b2)
{
	const auto leftIntersectionX = parabolasIntersectionX(y, a1, a2);
	const auto leftIntersectionY = parabolaY(y, a1, leftIntersectionX);

	const auto rightIntersectionX = parabolasIntersectionX(y, b1, b2);
	const auto rightIntersectionY = parabolaY(y, b1, rightIntersectionX);

	// y = m * x + f
	const auto m1 = (a1.x - a2.x) / (a2.y - a1.y);
	const auto f1 = -m1 * (a1.x + a2.x) / 2 + (a1.y + a2.y) / 2;

	const auto m2 = (b1.x - b2.x) / (b2.y - b1.y);
	const auto f2 = -m2 * (b1.x + b2.x) / 2 + (b1.y + b2.y) / 2;

	// TODO: isClose
	if (m1 == m2)
	{
		return false;
	}
	const auto intersectionX = (f2 - f1) / (m1 - m2);
	const auto intersectionY = m1 * intersectionX + f1;

	if (a1.x <= a2.x)
	{
		// It is left intersection, it goes to the left.
		if (intersectionX > leftIntersectionX)
			return false;
	}
	else
	{
		// It is right intersection, it goes to the right.
		if (intersectionX < leftIntersectionX)
			return false;
	}

	if (b1.x <= b2.x)
	{
		// It is left intersection, it goes to the left.
		if (intersectionX > rightIntersectionX)
			return false;
	}
	else
	{
		// It is right intersection, it goes to the right.
		if (intersectionX < rightIntersectionX)
			return false;
	}

	const auto site1 = a1;
	const auto site2 = a2;
	const auto site3 = ((b1.x == a1.x && b1.y == a1.y) || (b1.x == a2.x && b1.y == a2.y)) ? b2 : b1;
	if (y < get<0>(circleBottomPoint(site1, site2, site3)))
	{
		return false;
	}

	return true;
}

DoublyConnectedEdgeList fortune(const vector<Point>& points)
{
	auto dcel = DoublyConnectedEdgeList{};

	dcel.faces.resize(points.size());

	auto queue = PriorityQueue{ points };
	auto beachLine = BeachLine{ points };

	const auto createCircleEvents = [&points, &queue, &beachLine]
	(const double y,
		BeachLine::Node* left, BeachLine::Node* leftIntersection, BeachLine::Node* innerLeft,
		BeachLine::Node* innerRight, BeachLine::Node* rightIntersection, BeachLine::Node* right)
	{
		if(const auto leftleftIntersection = get<0>(beachLine.findIntersectionWithLeftLeaf(left)))
		{
			if(isConvergent(y, points[leftleftIntersection->p], points[leftleftIntersection->q], points[leftIntersection->p], points[leftIntersection->q]))
			{
				const auto leftleftSite = leftleftIntersection->p == left->p ? leftleftIntersection->q : leftleftIntersection->p;
				assert(leftleftSite != left->p && leftleftSite != innerLeft->p && left->p != innerLeft->p);
				const auto [bottom, center] = circleBottomPoint(points[leftleftSite], points[left->p], points[innerLeft->p]);
				left->circleEventId = queue.insertCircleEvent(bottom, center, left);
			}
		}
		if (const auto rightrightIntersection = get<0>(beachLine.findIntersectionWithRightLeaf(right)))
		{
			if(isConvergent(y, points[rightIntersection->p], points[rightIntersection->q], points[rightrightIntersection->p], points[rightrightIntersection->q]))
			{
				const auto rightrightSite = rightrightIntersection->p == right->p ? rightrightIntersection->q : rightrightIntersection->p;
				assert(rightrightSite != right->p && rightrightSite != innerRight->p && right->p != innerRight->p);
				const auto [bottom, center] = circleBottomPoint(points[rightrightSite], points[right->p], points[innerRight->p]);
				right->circleEventId = queue.insertCircleEvent(bottom, center, right);
			}
		}
	};

	if (!queue.empty())
	{
		const auto [ev, evId] = queue.pop();
		beachLine.insertArc(evId);
	}

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
			assert(central->p != left->p);
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

			setNextAndPrev( intersectionEdgepq, leftIntersectionEdgepq, rightIntersectionEdgepq, s1, s2 );
			setNextAndPrev( leftIntersectionEdgepq, rightIntersectionEdgepq, intersectionEdgepq, bads, s1 );
			setNextAndPrev( rightIntersectionEdgepq, intersectionEdgepq, leftIntersectionEdgepq, s2, bads );

			createCircleEvents(ev.y, left, intersection, right, left, intersection, right);
		}
		}
	}

	return dcel;
}

int main()
{
	const auto points = vector<Point>{ {0, 10}, {1, 9}, {5, 8}, {3, 4}, {4, 5}, {1,-1}, {5,-2}, {-5,-5}, {-10,-6}, {-9, 2}, {-11,7}, {-3, 0}, {-2,6} };
	const auto vor = fortune(points);
	const auto [minX, maxX, minY, maxY] = points.size() == 0
		? make_tuple(0.0, 0.0, 0.0, 0.0)
		: reduce(
			cbegin(points), cend(points),
			make_tuple(numeric_limits<double>::max(), numeric_limits<double>::min(), numeric_limits<double>::max(), numeric_limits<double>::min()),
			[](const auto& accum, const Point& p)
			{
				return make_tuple(min(get<0>(accum), p.x), max(get<1>(accum), p.x), min(get<2>(accum), p.y), max(get<3>(accum), p.y));
			}
			);
	constexpr auto regionSizeMultiplier = 1.61803398875;
	const auto regionCenterX = (minX + maxX) / 2;
	const auto regionCenterY = (minY + maxY) / 2;
	const auto regionHalfSizeExact = regionSizeMultiplier * max(maxX - minX, maxY - minY) / 2;
	const auto regionHalfSize = isClose(regionHalfSizeExact, 0.0, 0.0, 1e-50) ? 0.5 : regionHalfSizeExact;
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
			
			const auto calcIsDirLeft = [](const Point& s1, const Point& s2, const Point& bads)
			{
				// ax + by + c = 0
				const auto a = -(s2.y - s1.y) / (s2.x - s1.x);
				const auto b = 1.0;
				const auto c = -(s2.y + a * s2.x);

				const auto dist = (a * bads.x + b * bads.y + c);

				const auto middleX = (s2.x + s1.x) / 2;
				const auto middleY = (s2.y + s1.y) / 2;

				return (a * (middleX + 1) + b * middleY + c) * dist > 0;
			};
			const auto isDirLeft = calcIsDirLeft(points[s1], points[s2], points[bads]);

			// Ray: y = m * x + f
			const auto m = (points[s1].x - points[s2].x) / (points[s2].y - points[s1].y);
			const auto f = -m * (points[s1].x + points[s2].x) / 2 + (points[s1].y + points[s2].y) / 2;
			infEdgeAs.push_back(e.vertexFrom);
			infEdgeBXs.push_back(isDirLeft ? drawMinX : drawMaxX);
			infEdgeBYs.push_back(m * infEdgeBXs.back() + f);
		}
		else
		{
			assert(points.size() == 2);
			const auto s1 = e1.face;
			const auto s2 = e2.face;
			assert(s1 != s2);
			const auto m = (points[s1].x - points[s2].x) / (points[s2].y - points[s1].y);
			const auto f = -m * (points[s1].x + points[s2].x) / 2 + (points[s1].y + points[s2].y) / 2;
			doubleInfEdgeAXs.push_back(drawMinX);
			doubleInfEdgeAYs.push_back(m * drawMinX + f);
			doubleInfEdgeBXs.push_back(drawMaxX);
			doubleInfEdgeBYs.push_back(m * drawMaxX + f);
		}
	}
	assert(doubleInfEdgeAXs.size() <= 1);

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
			ax.plot(doubleInfEdgeXs, doubleInfEdgeYs)
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
