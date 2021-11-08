#include <vector>
#include <algorithm>
#include <tuple>
#include <limits>
#include <cassert>
#include <iostream>

using namespace std;

struct Point
{
	double x;
	double y;
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
	return x * x / (2 * p) - (site.x * x) / p + (site.x * site.x) / (2 * p) + p / 2;
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
	size_t pointIndToInsert;

	constexpr BeachLine(const std::vector<Point>& points)
		: points{ points }
		, root{ nullptr }
		, pointIndToInsert{ 0 }
	{
		assert(is_sorted(cbegin(points), cend(points), [](const auto& a, const auto& b) { return a.y > b.y; }));
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

	constexpr NewArcInfo insertArc()
	{
		const auto site = pointIndToInsert++;
		if (empty())
		{
			root = new Node{ .parent = nullptr, .left = nullptr, .right = nullptr, .p = 0, .q = 0, .circleEventId = -1, .balance = 0 };
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

		auto newSubTree = new Node{ .parent = regionNodeParent, .left = nullptr, .right = nullptr, .p = leftInd, .q = rightInd, .circleEventId = -1, .balance = 1 };
		newSubTree->left = regionNode;
		newSubTree->left->parent = newSubTree;
		newSubTree->left->circleEventId = -1;
		newSubTree->right = new Node{ .parent = newSubTree, .left = nullptr, .right = nullptr, .p = rightInd, .q = leftInd, .circleEventId = -1, .balance = 0 };
		newSubTree->right->left = new Node{ .parent = newSubTree->right, .left = nullptr, .right = nullptr, .p = site, .q = 0, .circleEventId = -1, .balance = 0 };
		newSubTree->right->right = new Node{ .parent = newSubTree->right, .left = nullptr, .right = nullptr, .p = intersectedArc, .q = 0, .circleEventId = -1, .balance = 0 };

		const auto res = NewArcInfo{
			.id = eventId,
			.left = newSubTree->left,
			.leftCentralIntersection = newSubTree,
			.central = newSubTree->right->left,
			.centralRightIntersection = newSubTree->right,
			.right = newSubTree->right->right
		};
		
		replaceNode(regionNode, newSubTree, regionNodeParent);

		auto child = newSubTree;
		bool needRebalance = false;
		if (regionNodeParent != nullptr)
		{
			if (regionNodeParent->left == newSubTree)
				regionNodeParent->balance -= 2;
			else
				regionNodeParent->balance += 2;
			assert(-2 <= regionNodeParent->balance && regionNodeParent->balance <= 2);
			child = regionNodeParent;
			regionNodeParent = regionNodeParent->parent;
			needRebalance = rebalanceSubTree(child, regionNodeParent);
		}

		while (regionNodeParent != nullptr && (needRebalance || child->balance == 0))
		{
			if (regionNodeParent->left == child)
				--regionNodeParent->balance;
			else
				++regionNodeParent->balance;

			child = regionNodeParent;
			regionNodeParent = regionNodeParent->parent;
			rebalanceSubTree(child, regionNodeParent);
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
		if (s1 > s2)
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

	constexpr static Node* findLeafToLeft(Node* node)
	{
		assert(node->isLeaf());
		auto [res, _] = findIntersectionWithLeftLeaf(node);
		if (res == nullptr)
			return nullptr;
		res = res->left;
		while (!res->isLeaf())
			res = res->right;
		return res;
	}

	constexpr static Node* findLeafToRight(Node* node)
	{
		assert(node->isLeaf());
		auto [res, _] = findIntersectionWithRightLeaf(node);
		if (res == nullptr)
			return nullptr;
		res = res->right;
		while (!res->isLeaf())
			res = res->left;
		return res;
	}

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

	constexpr void rebalanceSubTree(Node* node)
	{
		rebalanceSubTree(node, node->parent);
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

	constexpr void replaceNode(Node* oldNode, Node* newNode)
	{
		replaceNode(oldNode, newNode, oldNode->parent);
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
		assert(is_sorted(cbegin(points), cend(points), [](const auto& a, const auto& b) { return a.y > b.y; }));
		idToInd.reserve(points.size());
		for (size_t i{ 0 }; i != points.size(); ++i)
		{
			storage[i] = Event{ .y = points[i].y, .type = Event::Type::site };
			ids[i] = i;
			idToInd.push_back(i);
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

	constexpr Event pop()
	{
		assert(!empty());
		const auto res = storage[0];
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

bool isIntersectionBelow(const double y, const Point& a1, const Point& a2, const Point& b1, const Point& b2)
{
	// TODO: simplify computations? Direct formula for lines intersection?
	const auto anotherYA = y + a1.y;
	const auto anotherYB = y + b1.y;

	const auto lineA1x = parabolasIntersectionX(anotherYA, a1, a2);
	const auto lineA1y = parabolaY(anotherYA, a1, lineA1x);
	const auto lineA2x = parabolasIntersectionX(y, a1, a2);
	const auto lineA2y = parabolaY(y, a1, lineA2x);

	const auto lineB1x = parabolasIntersectionX(anotherYB, b1, b2);
	const auto lineB1y = parabolaY(anotherYB, b1, lineB1x);
	const auto lineB2x = parabolasIntersectionX(y, b1, b2);
	const auto lineB2y = parabolaY(y, b1, lineB2x);

	const auto lineAdx = lineA2x - lineA1x;
	const auto lineAdy = lineA2y - lineA1y;
	const auto lineBdx = lineB2x - lineB1x;
	const auto lineBdy = lineB2y - lineB1y;

	const auto numer = lineBdy * (lineA1y * lineA2x - lineA1x * lineA2y) + lineAdy * (lineB1x * lineB2y - lineB1y * lineB2x);
	const auto denom = lineAdx * lineBdy - lineBdx * lineAdy;

	return numer / denom <= y;
}

struct Edge
{
	size_t i;
	size_t j;
};

struct VoronoiDiagram
{
	vector<Point> vertices;
	vector<Edge> edges;
};

VoronoiDiagram fortune(vector<Point> points)
{
	auto res = VoronoiDiagram{};

	sort(begin(points), end(points), [](const auto& a, const auto& b) { return a.y > b.y; });
	auto queue = PriorityQueue{ points };
	auto beachLine = BeachLine{ points };

	const auto createCircleEvents = [&points, &queue, &beachLine]
	(const double y,
		BeachLine::Node* left, BeachLine::Node* leftIntersection, BeachLine::Node* innerLeft,
		BeachLine::Node* innerRight, BeachLine::Node* rightIntersection, BeachLine::Node* right)
	{
		if(const auto leftleftIntersection = get<0>(beachLine.findIntersectionWithLeftLeaf(left)))
		{
			if (isIntersectionBelow(y, points[leftleftIntersection->p], points[leftleftIntersection->q], points[leftIntersection->p], points[leftIntersection->q]))
			{
				const auto leftleftSite = leftleftIntersection->p == left->p ? leftleftIntersection->q : leftleftIntersection->p;
				assert(leftleftSite != left->p && leftleftSite != innerLeft->p && left->p != innerLeft->p);
				const auto [bottom, center] = circleBottomPoint(points[leftleftSite], points[left->p], points[innerLeft->p]);
				left->circleEventId = queue.insertCircleEvent(bottom, center, left);
			}
		}
		if (const auto rightrightIntersection = get<0>(beachLine.findIntersectionWithRightLeaf(right)))
		{
			if (isIntersectionBelow(y, points[rightIntersection->p], points[rightIntersection->q], points[rightrightIntersection->p], points[rightrightIntersection->q]))
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
		queue.pop();
		beachLine.insertArc();
	}

	while (!queue.empty())
	{
		auto ev = queue.pop();
		cout << ev.y << "\n";
		switch (ev.type)
		{
		break;  case Event::Type::site:
		{
			const auto [eventId, left, leftCentral, central, centralRight, right] = beachLine.insertArc();
			if (eventId != -1)
			{
				queue.removeById(eventId);
			}
			createCircleEvents(ev.y, left, leftCentral, central, central, centralRight, right);
		}
		break; case Event::Type::circle:
		{
			const auto arcToRemove = ev.leaf;
			const auto left = beachLine.findLeafToLeft(arcToRemove);
			const auto right = beachLine.findLeafToRight(arcToRemove);
			assert(left != nullptr);
			assert(right != nullptr);
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
			const auto intersection = beachLine.removeArc(arcToRemove);
			createCircleEvents(ev.y, left, intersection, right, left, intersection, right);
		}
		}
	}

	return res;
}

int main()
{
	auto vor = fortune({ {0, 10}, {1, 9}, {5, 8}, {0.5, 7} });
	return 0;
}
