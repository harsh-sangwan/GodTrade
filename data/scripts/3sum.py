import unittest

class Solution(object):

	def check(self, nums, n):
		if len(nums) < 3:
			print 'List has less than 3 elements'
			return False

		if any(not isinstance(x, int) for x in nums):
			print 'Non integers in list'
			return False

		if type(n) != int:
			print 'n is not an integer'
			return False

		return True

	#nums -> list of numbers ; n -> number = three numbers, only integers are accepted
	def three_sum(self, nums, n):

		#check if valid -> all numbers in nums are int and n is int
		if(self.check(nums, n)):

			nums = sorted(nums)
			for i in range(0, (len(nums)-2)):

				j = i+1
				k = len(nums)-1

				while j < k:

					if((nums[i] + nums[j] + nums[k]) == n):
						print nums[i], nums[j], nums[k]
						return True

					elif((nums[i] + nums[j] + nums[k]) < n):
						j = j+1

					else:
						k = k-1

			print 'No three numbers found'
			return False

		return False

Ns = [[1, 4, 3, 6, 7, 10, 13, 54], [1, 4], [1, 4, 3, -6, 7, -10, 13, 54], [1, 4, -3.2, 6, 7, -10.2, 13.1, -54], 
[1, 4, 3, 6, 7, 'orange', 13, 54], [1, 4, 3, 6, 7, 10, 13, 54], [1, 4, 3, 6, 7, 10, 13, 54], [-1, -4, -3, 6, -7, 10, -13, 54]]

ns = [12, 12, 12, 12, 12, 'apples', 12.12, -12]

if __name__ == "__main__":

	obj = Solution()

	'''#Test cases - OK
	nums = [1, 4, 3, 6, 7, 10, 13, 54]
	n = 12

	print obj.three_sum(nums, n)

	#Test cases - less than 3 ints
	nums = [1, 4]
	n = 12

	print obj.three_sum(nums, n)

	#Test cases - negative nums - OK
	nums = [1, 4, 3, -6, 7, -10, 13, 54]
	n = 12

	print obj.three_sum(nums, n)

	#Test cases - float 
	nums = [1, 4, -3.2, 6, 7, -10.2, 13.1, -54]
	n = 12

	print obj.three_sum(nums, n)

	#Test cases - strings in between
	nums = [1, 4, 3, 6, 7, 'orange', 13, 54]
	n = 12

	print obj.three_sum(nums, n)

	#Test cases - n is string
	nums = [1, 4, 3, 6, 7, 10, 13, 54]
	n = 'apples'

	print obj.three_sum(nums, n)

	#Test cases n is float
	nums = [1, 4, 3, 6, 7, 10, 13, 54]
	n = 12.12

	print obj.three_sum(nums, n)

	#Test cases negative
	nums = [-1, -4, -3, 6, -7, 10, -13, 54]
	n = -12

	print obj.three_sum(nums, n)'''