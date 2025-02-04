// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EscrowContract {
    address public payer;
    address public payee;
    uint public amount;
    bool public isApproved;

    event EscrowCreated(address indexed payer, address indexed payee, uint amount);
    event PaymentReleased(address indexed payee, uint amount);

    constructor(address _payee, uint _amount) {
        payer = msg.sender;
        payee = _payee;
        amount = _amount;
        isApproved = false;
        emit EscrowCreated(payer, payee, amount);
    }

    function approveEscrow() public {
        require(msg.sender == payer, "Only payer can approve escrow.");
        isApproved = true;
    }

    function releasePayment() public {
        require(isApproved, "Escrow not approved.");
        require(msg.sender == payer, "Only payer can release payment.");
        payable(payee).transfer(amount);
        emit PaymentReleased(payee, amount);
    }

    // Fallback function to receive funds.
    receive() external payable {}
}
