import numpy as np
import pickle
import matplotlib.pyplot as plt

exp_p1 = list()  # список результатов для p1
exp_p2 = list()  # список результатов для p2

class State:
    """
    Класс обучения двух ботов
    p1 - делает первый ход
    """
    def __init__(self, p1, p2):
        self.board = np.zeros((3, 3)) # создание доски
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False # флаг для раунда
        self.boardHash = None
        self.playerSymbol = 1 # p1 делает первый ход

    def getHash(self):
        #Функция хеширования состояние поля
        self.boardHash = str(self.board.reshape(3 * 3))
        return self.boardHash

    def winner(self):
        # проверяем строки
        for i in range(3):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # проверяем столбцы
        for i in range(3):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # проверяем диагонали
        diag_sum1 = sum([self.board[i, i] for i in range(3)])
        diag_sum2 = sum([self.board[i, 3 - i - 1] for i in range(3)])
        diag_sum = max(diag_sum1, diag_sum2, key=abs)
        if diag_sum == 3:
            self.isEnd = True
            return 1
        if diag_sum == -3:
            self.isEnd = True
            return -1

        # ничья
        # нет доступных позиций
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # игра не закончена
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # переключение на другого игрока
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # только если игра закончена
    def giveReward(self):
        result = self.winner()
        # вознаграждение
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
            exp_p1.append(1)
            exp_p2.append(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
            exp_p1.append(0)
            exp_p2.append(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)
            exp_p1.append(0.1)
            exp_p2.append(0.5)

    # обновление поля
    def reset(self):
        self.board = np.zeros((3, 3))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Количество итераций: {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # выполняем действие и обновляем состояние поля
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                # проверяем, не закончилась ли игра
                win = self.winner()
                if win is not None:
                    #self.showBoard()
                    # игра закончилась победой игрока p1 или ничьей
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # игра закончилась победой игрока p2 или ничьей
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, 3):
            print('-------------')
            out = '| '
            for j in range(0, 3):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


class Player:
    def __init__(self, name, exp_rate=0.3):

        #V(S) = V(s) + A * [V(s)' - V(s)]
        self.name = name
        self.states = []  # записываем все занятые позиции
        self.lr = 0.2 # A
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # состояние -> значение

    def getHash(self, board):
        boardHash = str(board.reshape(3 * 3))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # выбираем рандомное действие
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    # добавляем сохранённое состояние
    def addState(self, state):
        self.states.append(state)

    # в конце игры обновляем значения состояний в обратном порядке
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('exp_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


def show_stats(exp_p1, exp_p2, rounds):
    x = np.arange(rounds)
    y1 = sorted(np.array(exp_p1))
    y2 = sorted(np.array(exp_p2))

    plt.xticks(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()

p1 = Player("p1")
p2 = Player("p2")

#st = State(p1, p2)
rounds = 1000
#st.play(rounds)

#p1.savePolicy()
#p2.savePolicy()
#
p1.loadPolicy("exp_p1")
p2.loadPolicy("exp_p2_100")
st = State(p1, p2)
st.play(rounds)
show_stats(exp_p1, exp_p2, rounds)

