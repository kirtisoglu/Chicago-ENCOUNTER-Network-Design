
def merge_sort(pair_list):

    if len(pair_list) > 1:

        mid = len(pair_list) // 2
        left_half = pair_list[:mid]
        right_half = pair_list[mid:]

        # Recursively sort each half
        merge_sort(left_half)
        merge_sort(right_half)

        # Merge the sorted halves
        i, j, k = 0, 0, 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i][1] < right_half[j][1]:
                pair_list[k] = left_half[i]
                i += 1
            else:
                pair_list[k] = right_half[j]
                j += 1
            k += 1

        # Check for any remaining elements in left_half
        while i < len(left_half):
            pair_list[k] = left_half[i]
            i += 1
            k += 1

        # Check for any remaining elements in right_half
        while j < len(right_half):
            pair_list[k] = right_half[j]
            j += 1
            k += 1


def sort_dictionary(dictionary):

    list = [(key, dictionary[key]) for key in dictionary.keys()]
    merge_sort(list)

    return list

