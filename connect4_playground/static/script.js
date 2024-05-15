document.addEventListener('DOMContentLoaded', () => {
    const gameBoard = document.getElementById('game-board');
    const currentPlayerSpan = document.getElementById('current-player');
    const gameStatusSpan = document.getElementById('game-status');
    const epsilonSlider = document.getElementById('epsilon-slider');
    const epsilonValueSpan = document.getElementById('epsilon-value');
    const resetButton = document.getElementById('reset-button');

    let board = Array(6).fill(0).map(() => Array(7).fill(0));
    let currentPlayer = 1;
    let gameOver = false;

    // Create game board
    createBoard();

    // Event listeners
    gameBoard.addEventListener('click', handleCellClick);
    epsilonSlider.addEventListener('input', updateEpsilon);
    resetButton.addEventListener('click', resetGame);

    function createBoard() {
        for (let r = 0; r < 6; r++) {
            for (let c = 0; c < 7; c++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = r;
                cell.dataset.col = c;
                gameBoard.appendChild(cell);
            }
        }
    }

    async function handleCellClick(event) {
        if (gameOver) return;
        const row = parseInt(event.target.dataset.row);
        const col = parseInt(event.target.dataset.col);

        if (isValidMove(col)) {
            dropPiece(row, col, currentPlayer);
            if (checkWin(row, col, currentPlayer)) {
                endGame(`Player ${currentPlayer} wins!`);
            } else if (checkDraw()) {
                endGame("It's a draw!");
            } else {
                switchPlayer();
                await getAIMove();
            }
        }
    }

    function isValidMove(col) {
        return board[0][col] === 0;
    }

    function dropPiece(row, col, player) {
        board[row][col] = player;
        const cell = getCell(row, col);
        cell.classList.add(`player${player}`);
    }

    async function getAIMove() {
        const response = await fetch('/get_ai_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                board: board,
                player: currentPlayer,
                epsilon: parseFloat(epsilonSlider.value)
            })
        });
        const data = await response.json();
        const aiCol = data.column;

        if (isValidMove(aiCol)) {
            const aiRow = getLowestEmptyRow(aiCol);
            dropPiece(aiRow, aiCol, currentPlayer);
            if (checkWin(aiRow, aiCol, currentPlayer)) {
                endGame(`Player ${currentPlayer} wins!`);
            } else if (checkDraw()) {
                endGame("It's a draw!");
            } else {
                switchPlayer();
            }
        } else {
            console.error("Invalid AI move:", aiCol); // Handle invalid moves (optional)
        }
    }

    function switchPlayer() {
        currentPlayer = 3 - currentPlayer;
        currentPlayerSpan.textContent = currentPlayer;
    }

    function getLowestEmptyRow(col) {
        for (let r = 5; r >= 0; r--) {
            if (board[r][col] === 0) {
                return r;
            }
        }
        return -1; // Column is full (shouldn't happen if isValidMove is checked)
    }

    function getCell(row, col) {
        return gameBoard.querySelector(`.cell[data-row="<span class="math-inline">\{row\}"\]\[data\-col\="</span>{col}"]`);
    }

    function checkWin(row, col, player) {
        const directions = [
            [1, 0], [0, 1], [1, 1], [1, -1]
        ];

        for (const [dr, dc] of directions) {
            let count = 1;
            for (let step = 1; step < 4; step++) {
                const newRow = row + step
