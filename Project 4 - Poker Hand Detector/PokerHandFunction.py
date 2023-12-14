def findPokerHand(hand):
    ranks = [] # значения карт, например 2, 7 или Q
    suits = [] # масти карт
    possibleRanks = [] # возможные комбинации, например, 10 - Роял Флеш, 9 - Стрит Флеш и т.д.

    # карта состоит из значения (card[0] или card[0:2]) и масти (card[1] или card[2])
    for card in hand:
        # фиксируем значение и масть
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
        # если значение - буква, то переводим его в число
        if rank == "A":
            rank = 14
        elif rank == "K":
            rank = 13
        elif rank == "Q":
            rank = 12
        elif rank == "J":
            rank = 11
        # добавляем в список значений карт и список мастей
        ranks.append(int(rank))
        suits.append(suit)

    # сортируем значения карт
    sortedRanks = sorted(ranks)

    # Роял Флеш, Стрит Флеш и просто Флеш
    if suits.count(suits[0]) == 5: # проверка на Флеш
        # проверка на Роял
        if 14 in sortedRanks and 13 in sortedRanks and 12 in sortedRanks and 11 in sortedRanks \
                and 10 in sortedRanks:
            possibleRanks.append(10)
        # проверка на Стрит
        elif all(sortedRanks[i] == sortedRanks[i - 1] + 1 for i in range(1, len(sortedRanks))):
            possibleRanks.append(9)
        else:
            possibleRanks.append(6) # если не Роял и не Стрит, то просто Флеш

    # Стрит
    if all(sortedRanks[i] == sortedRanks[i - 1] + 1 for i in range(1, len(sortedRanks))):
        possibleRanks.append(5)

    # фиксируем уникальные значения карт на руке
    handUniqueVals = list(set(sortedRanks))

    # Каре и Фулл Хаус
    # 3 3 3 3 5   -- set --- 3 5 --- unique values = 2 --- Каре
    # 3 3 3 5 5   -- set -- 3 5 ---- unique values = 2 --- Фулл Хаус
    if len(handUniqueVals) == 2:
        for val in handUniqueVals:
            if sortedRanks.count(val) == 4:  # --- Каре
                possibleRanks.append(8)
            if sortedRanks.count(val) == 3:  # --- Фулл Хаус
                possibleRanks.append(7)

    # Сет и 2 пары
    # 5 5 5 6 7 -- set -- 5 6 7 --- unique values = 3   -- Сет
    # 8 8 7 7 2 -- set -- 8 7 2 --- unique values = 3   -- 2 пары
    if len(handUniqueVals) == 3:
        for val in handUniqueVals:
            if sortedRanks.count(val) == 3:  # -- Сет
                possibleRanks.append(4)
            if sortedRanks.count(val) == 2:  # -- 2 пары
                possibleRanks.append(3)

    # 1 пара
    # 5 5 3 6 7 -- set -- 5 3 6 7 - unique values = 4 -- 1 пара
    if len(handUniqueVals) == 4:
        possibleRanks.append(2)

    # если до этого не было комбинаций, то остаётся только старшая карта
    if not possibleRanks:
        possibleRanks.append(1)

    # словарь названий комбинаций
    pokerHandRanks = {10: "Royal Flush", 9: "Straight Flush", 8: "Four of a Kind", 7: "Full House", 6: "Flush",
                      5: "Straight", 4: "Three of a Kind", 3: "Two Pair", 2: "Pair", 1: "High Card"}
    # выбрали максимальную комбинацию
    output = pokerHandRanks[max(possibleRanks)]
    # вывели руку и максимальную комбинацию
    print(hand, output)
    return output


if __name__ == "__main__":
    findPokerHand(["KH", "AH", "QH", "JH", "10H"])  # Роял Флеш
    findPokerHand(["QC", "JC", "10C", "9C", "8C"])  # Стрит Флеш
    findPokerHand(["5C", "5S", "5H", "5D", "QH"])  # Каре
    findPokerHand(["2H", "2D", "2S", "10H", "10C"])  # Фулл Хайс
    findPokerHand(["2D", "KD", "7D", "6D", "5D"])  # Флеш
    findPokerHand(["JC", "10H", "9C", "8C", "7D"])  # Стрит
    findPokerHand(["10H", "10C", "10D", "2D", "5S"])  # Сет
    findPokerHand(["KD", "KH", "5C", "5S", "6D"])  # 2 пары
    findPokerHand(["2D", "2S", "9C", "KD", "10C"])  # 1 пара
    findPokerHand(["KD", "5H", "2D", "10C", "JH"])  # Старшая карта
